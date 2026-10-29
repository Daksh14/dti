//! The training worker. It runs inside `dti-runner`, which grants it outbound
//! HTTP to the coordinator and the volunteer's data origins, `wasi:nn` for
//! the TorchScript graphs the volunteer configured, and `dti:runner/telemetry`.
//!
//! Each round follows DiLoCo: pull the global weights, run `inner_steps` of
//! AdamW on the next text in this worker's slices of the data, then push
//! `global - local` (int8-compressed) back to the coordinator. The forward and
//! backward passes run in libtorch through `wasi:nn`. The worker stops once it
//! has downloaded as much data as the volunteer's budget allows.

mod data;
mod http;
mod nn;
mod parquet;

wit_bindgen::generate!({
    path: "../../wit",
    world: "worker",
    generate_all,
});

use data::Data;
use dti::runner::telemetry;
use dti_model::AdamW;
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
    let usage = "usage: dti-worker <coordinator-url> <data-budget-bytes>";
    let mut args = std::env::args().skip(1);
    let (Some(base), Some(budget)) = (args.next(), args.next()) else {
        return Err(usage.into());
    };
    let budget: u64 = budget.parse().map_err(|_| usage)?;
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
    telemetry::log(&format!("training on {}", run.data));

    let num_params = usize::try_from(run.num_params).map_err(|_| "model too large")?;
    let graph = nn::Graph::load(&run.model.to_string(), num_params)
        .map_err(|e| format!("{e} (the runner config needs it under [graphs])"))?;
    let mut data = Data::new(&client, &worker_id, budget);
    let mut opt = AdamW::new(num_params, run.train.weight_decay);
    let mut grad = vec![0.0; num_params];
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

        let round = Round {
            client: &client,
            worker_id: &worker_id,
            task: &task,
            train: &run.train,
            graph: &graph,
        };
        match round.train(&mut data, &mut opt, &mut grad) {
            Ok(true) => {
                failures = 0;
                last_round = Some(task.round);
            }
            Ok(false) => {
                telemetry::log(&if data.budget_left() == 0 {
                    "data budget used up".to_string()
                } else {
                    format!(
                        "no more slices to train on that fit the {:.3} GB left of the budget",
                        data.budget_left() as f64 / 1e9
                    )
                });
                return Ok(());
            }
            Err(e) => {
                // Train this round's text again on the next attempt.
                data.rewind();
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
    }
}

struct Round<'a> {
    client: &'a Client,
    worker_id: &'a str,
    task: &'a Task,
    train: &'a dti_protocol::TrainConfig,
    graph: &'a nn::Graph,
}

impl Round<'_> {
    /// Trains one round and sends the update. `Ok(false)` if there was no
    /// data left to train on.
    fn train(&self, data: &mut Data, opt: &mut AdamW, grad: &mut [f32]) -> Result<bool, String> {
        let (client, round) = (self.client, self.task.round);
        let weights = client.get(&format!("/v1/weights/{round}"))?;
        if weights.status == 409 {
            return Err("round moved on before we fetched weights".into());
        }
        expect_status(&weights, 200, "weights")?;
        let global = decode_f32s(&weights.body).map_err(|e| format!("weights: {e}"))?;
        if global.len() != self.graph.num_params() {
            return Err(format!(
                "weights: expected {} params, got {}",
                self.graph.num_params(),
                global.len()
            ));
        }

        let mut params = global.clone();
        let inner_steps = self.train.inner_steps;
        let report_every = (inner_steps / 10).max(1);
        let mut tokens = 0u64;
        let mut steps = 0;
        while steps < inner_steps {
            let Some(batch) = data.next_batch(self.train.batch_tokens as usize)? else {
                break;
            };
            let loss = self.graph.loss_and_grad(&params, batch, grad)?;
            opt.step(&mut params, grad, self.train.inner_lr);
            tokens += batch.len() as u64;
            steps += 1;
            if (steps - 1) % report_every == 0 || steps == inner_steps {
                telemetry::step(telemetry::StepMetrics {
                    round,
                    step: steps,
                    inner_steps,
                    loss,
                });
            }
        }
        if steps == 0 {
            return Ok(false);
        }

        let delta: Vec<f32> = global.iter().zip(&params).map(|(g, l)| g - l).collect();
        let body = encode_update(&delta);
        let sent = client.post(
            &format!("/v1/update/{}/{round}?tokens={tokens}", self.worker_id),
            "application/octet-stream",
            &body,
        )?;
        if sent.status == 409 {
            return Err("update arrived after the round closed".into());
        }
        expect_status(&sent, 202, "update")?;
        telemetry::round_finished(round, body.len() as u64);

        let done = data.commit();
        if !done.is_empty() {
            let ids = serde_json::to_vec(&done).map_err(|e| e.to_string())?;
            // If this is lost, the coordinator leases the slices again later.
            match client.post(
                &format!("/v1/done/{}", self.worker_id),
                "application/json",
                &ids,
            ) {
                Ok(r) if r.status == 204 => {}
                Ok(r) => telemetry::log(&format!("reporting finished slices: HTTP {}", r.status)),
                Err(e) => telemetry::log(&format!("reporting finished slices: {e}")),
            }
        }
        Ok(true)
    }
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
