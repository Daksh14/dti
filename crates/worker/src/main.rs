//! The training worker. It runs inside `dti-runner`, which grants it outbound
//! HTTP to the coordinator and the `dti:runner/telemetry` interface, and nothing else.
//!
//! Each round follows DiLoCo: pull the global weights, run `inner_steps` of
//! AdamW on this worker's shard, then push `global - local` (int8-compressed)
//! back to the coordinator.

mod http;

wit_bindgen::generate!({
    path: "../../wit",
    world: "worker",
});

use dti::runner::telemetry;
use dti_model::{AdamW, Rng};
use dti_protocol::{JoinResponse, Task, decode_f32s, encode_update};
use http::Client;

/// Consecutive failed rounds before the worker gives up.
const MAX_FAILURES: u32 = 10;

fn main() {
    if let Err(e) = run() {
        telemetry::log(&format!("fatal: {e}"));
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let base = std::env::args()
        .nth(1)
        .ok_or("usage: dti-worker <coordinator-url>")?;
    let client = Client::new(&base)?;

    let joined = client.post("/v1/join", "application/json", b"{}")?;
    expect_status(&joined, 200, "join")?;
    let JoinResponse { worker_id, run } =
        serde_json::from_slice(&joined.body).map_err(|e| format!("join: {e}"))?;
    telemetry::joined(&telemetry::RunInfo {
        run_id: run.run_id.clone(),
        model: run.model.to_string(),
        params: run.num_params,
        worker_id: worker_id.clone(),
    });

    let model = dti_model::build(run.model);
    let mut opt = AdamW::new(model.num_params(), run.train.weight_decay);
    let mut grad = vec![0.0; model.num_params()];
    let mut last_round = None;
    let mut failures = 0;

    loop {
        let task = match client.get(&format!("/v1/task/{worker_id}"))? {
            r if r.status == 410 => {
                telemetry::log("run finished");
                return Ok(());
            }
            r => {
                expect_status(&r, 200, "task")?;
                serde_json::from_slice::<Task>(&r.body).map_err(|e| format!("task: {e}"))?
            }
        };
        if last_round.is_some_and(|r| task.round <= r) {
            // Our update for this round is in; wait for the others.
            sleep_ms(500);
            continue;
        }

        match train_round(
            &client, &worker_id, &task, &run.train, &*model, &mut opt, &mut grad,
        ) {
            Ok(()) => failures = 0,
            Err(e) => {
                failures += 1;
                telemetry::log(&format!(
                    "round {} failed ({failures}/{MAX_FAILURES}): {e}",
                    task.round
                ));
                if failures >= MAX_FAILURES {
                    return Err("too many consecutive failures".into());
                }
                sleep_ms(2000);
            }
        }
        last_round = Some(task.round);
    }
}

fn train_round(
    client: &Client,
    worker_id: &str,
    task: &Task,
    train: &dti_protocol::TrainConfig,
    model: &dyn dti_model::Model,
    opt: &mut AdamW,
    grad: &mut [f32],
) -> Result<(), String> {
    let weights = client.get(&format!("/v1/weights/{}", task.round))?;
    if weights.status == 409 {
        return Err("round moved on before we fetched weights".into());
    }
    expect_status(&weights, 200, "weights")?;
    let global = decode_f32s(&weights.body).map_err(|e| format!("weights: {e}"))?;
    if global.len() != model.num_params() {
        return Err(format!(
            "weights: expected {} params, got {}",
            model.num_params(),
            global.len()
        ));
    }

    let data = client.get(&format!(
        "/v1/data/{}/{}",
        task.shard.offset, task.shard.len
    ))?;
    expect_status(&data, 200, "data")?;
    let data = data.body;
    let window = (train.batch_tokens as usize).min(data.len());
    if window < 2 {
        return Err("shard too small to train on".into());
    }

    let mut params = global.clone();
    let mut rng = Rng::new(task.seed);
    let report_every = (train.inner_steps / 10).max(1);
    for step in 0..train.inner_steps {
        let start = rng.below((data.len() - window + 1) as u64) as usize;
        let loss = model.loss_and_grad(&params, &data[start..start + window], grad);
        opt.step(&mut params, grad, train.inner_lr);
        if step % report_every == 0 || step + 1 == train.inner_steps {
            telemetry::step(telemetry::StepMetrics {
                round: task.round,
                step: step + 1,
                inner_steps: train.inner_steps,
                loss,
            });
        }
    }

    let delta: Vec<f32> = global.iter().zip(&params).map(|(g, l)| g - l).collect();
    let body = encode_update(&delta);
    let sent = client.post(
        &format!("/v1/update/{worker_id}/{}", task.round),
        "application/octet-stream",
        &body,
    )?;
    if sent.status == 409 {
        return Err("update arrived after the round closed".into());
    }
    expect_status(&sent, 202, "update")?;
    telemetry::round_finished(task.round, body.len() as u64);
    Ok(())
}

fn expect_status(r: &http::Response, want: u16, what: &str) -> Result<(), String> {
    if r.status == want {
        return Ok(());
    }
    let msg = String::from_utf8_lossy(&r.body);
    Err(format!("{what}: HTTP {} {}", r.status, msg.trim()))
}

fn sleep_ms(ms: u64) {
    wasip2::clocks::monotonic_clock::subscribe_duration(ms * 1_000_000).block();
}
