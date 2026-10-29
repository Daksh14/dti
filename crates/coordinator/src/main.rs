//! Coordinates a DiLoCo training run: hands out data shards, collects
//! compressed pseudo-gradients from workers, combines them robustly and
//! applies the outer optimiser step.
//!
//! This is a single trusted coordinator. Splitting it across independent
//! aggregators that agree on each round's result is future work.

mod aggregate;

use aggregate::{Nesterov, l2, robust_mean};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use dti_model::Model;
use dti_protocol::{
    JoinResponse, ModelSpec, RunInfo, Shard, Task, TrainConfig, decode_update, encode_f32s,
    update_len,
};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Largest slice of the corpus a worker can fetch in one request.
const MAX_DATA_REQUEST: u64 = 16 << 20;
/// How much held-out text the validation loss is measured on.
const VAL_BYTES: usize = 64 << 10;

#[derive(Parser)]
#[command(version)]
struct Cli {
    /// Training text. The model is byte-level, so any file works.
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    /// Close a round as soon as this many workers have sent updates.
    #[arg(long, default_value_t = 2)]
    min_workers: usize,
    /// Close a round after this long if at least one update has arrived.
    #[arg(long, default_value_t = 120)]
    round_timeout_secs: u64,
    #[arg(long, default_value_t = 30)]
    rounds: u64,
    #[arg(long, default_value_t = 100)]
    inner_steps: u32,
    #[arg(long, default_value_t = 512)]
    batch_tokens: u32,
    #[arg(long, default_value_t = 0.02)]
    inner_lr: f32,
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,
    #[arg(long, default_value_t = 0.7)]
    outer_lr: f32,
    /// DiLoCo uses 0.9 for transformers; the toy bigram model converges better with less.
    #[arg(long, default_value_t = 0.5)]
    outer_momentum: f32,
    /// Fraction of updates dropped from each end, per coordinate.
    #[arg(long, default_value_t = 0.2)]
    trim_fraction: f32,
    #[arg(long, default_value_t = 64 << 10)]
    shard_bytes: u64,
    /// Write the final weights (raw little-endian f32) here when the run ends.
    #[arg(long)]
    out: Option<PathBuf>,
}

struct Coordinator {
    run: RunInfo,
    model: Box<dyn Model>,
    train: Arc<[u8]>,
    val: Vec<u8>,
    shard_bytes: u64,
    min_workers: usize,
    rounds: u64,
    trim_fraction: f32,
    round_timeout: Duration,
    out: Option<PathBuf>,

    params: Vec<f32>,
    outer: Nesterov,
    /// `params` encoded for `GET /v1/weights`, rebuilt once per round.
    weights: Bytes,
    round: u64,
    round_started: Instant,
    /// Worker id → index, used to spread workers over different shards.
    workers: HashMap<String, u64>,
    updates: BTreeMap<String, Vec<f32>>,
    val_loss: f32,
    finished: bool,
}

type Shared = Arc<Mutex<Coordinator>>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let corpus = std::fs::read(&cli.corpus)?;
    anyhow::ensure!(corpus.len() >= 1024, "corpus is too small to train on");
    let (train, val) = split_corpus(&corpus);
    let (train, val) = (train.as_slice(), val.as_slice());

    let spec = ModelSpec::ByteBigram;
    let model = dti_model::build(spec);
    let params = model.init();
    let run_id = format!(
        "run-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    );
    let mut coord = Coordinator {
        run: RunInfo {
            run_id,
            model: spec,
            num_params: model.num_params() as u64,
            train: TrainConfig {
                inner_steps: cli.inner_steps,
                batch_tokens: cli.batch_tokens,
                inner_lr: cli.inner_lr,
                weight_decay: cli.weight_decay,
            },
        },
        val_loss: model.loss(&params, val),
        outer: Nesterov::new(params.len(), cli.outer_lr, cli.outer_momentum),
        weights: Bytes::from(encode_f32s(&params)),
        params,
        model,
        train: train.into(),
        val: val.to_vec(),
        shard_bytes: cli.shard_bytes.max(2),
        min_workers: cli.min_workers.max(1),
        rounds: cli.rounds,
        trim_fraction: cli.trim_fraction,
        round_timeout: Duration::from_secs(cli.round_timeout_secs),
        out: cli.out,
        round: 0,
        round_started: Instant::now(),
        workers: HashMap::new(),
        updates: BTreeMap::new(),
        finished: false,
    };
    coord.finished = coord.rounds == 0;

    println!(
        "{}: model {} ({} params), {} train / {} val bytes, {} rounds × {} inner steps",
        coord.run.run_id,
        coord.run.model,
        coord.run.num_params,
        coord.train.len(),
        coord.val.len(),
        coord.rounds,
        coord.run.train.inner_steps,
    );
    println!("round 0: val loss {:.4}", coord.val_loss);

    let max_body = update_len(coord.params.len()) + 1024;
    let state: Shared = Arc::new(Mutex::new(coord));
    tokio::spawn(close_stale_rounds(state.clone()));

    let app = Router::new()
        .route("/v1/join", post(join))
        .route("/v1/task/{worker}", get(task))
        .route("/v1/weights/{round}", get(weights))
        .route("/v1/data/{offset}/{len}", get(data))
        .route("/v1/update/{worker}/{round}", post(update))
        .route("/v1/status", get(status))
        .layer(DefaultBodyLimit::max(max_body))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(cli.listen).await?;
    println!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn join(State(state): State<Shared>) -> Json<JoinResponse> {
    let mut c = state.lock().unwrap();
    let index = c.workers.len() as u64;
    let worker_id = format!("w{}", index + 1);
    c.workers.insert(worker_id.clone(), index);
    println!("{worker_id} joined ({} workers)", c.workers.len());
    Json(JoinResponse {
        worker_id,
        run: c.run.clone(),
    })
}

async fn task(State(state): State<Shared>, Path(worker): Path<String>) -> Response {
    let c = state.lock().unwrap();
    let Some(&index) = c.workers.get(&worker) else {
        return (StatusCode::NOT_FOUND, "unknown worker").into_response();
    };
    if c.finished {
        return (StatusCode::GONE, "run finished").into_response();
    }
    let num_shards = (c.train.len() as u64 / c.shard_bytes).max(1);
    // Different workers get different shards within a round; the offset
    // rotates which shard each worker sees from round to round.
    let shard = (index + c.round * 7919) % num_shards;
    let offset = shard * c.shard_bytes;
    let len = if shard + 1 == num_shards {
        c.train.len() as u64 - offset
    } else {
        c.shard_bytes
    };
    Json(Task {
        round: c.round,
        shard: Shard { offset, len },
        seed: splitmix(c.round << 32 | index),
    })
    .into_response()
}

async fn weights(State(state): State<Shared>, Path(round): Path<u64>) -> Response {
    let c = state.lock().unwrap();
    if round != c.round {
        return (
            StatusCode::CONFLICT,
            format!("current round is {}", c.round),
        )
            .into_response();
    }
    (
        [(header::CONTENT_TYPE, "application/octet-stream")],
        c.weights.clone(),
    )
        .into_response()
}

async fn data(State(state): State<Shared>, Path((offset, len)): Path<(u64, u64)>) -> Response {
    let train = state.lock().unwrap().train.clone();
    let end = offset
        .checked_add(len)
        .filter(|&end| end <= train.len() as u64);
    match end {
        Some(end) if len <= MAX_DATA_REQUEST => (
            [(header::CONTENT_TYPE, "application/octet-stream")],
            train[offset as usize..end as usize].to_vec(),
        )
            .into_response(),
        _ => (
            StatusCode::RANGE_NOT_SATISFIABLE,
            "range outside the training data",
        )
            .into_response(),
    }
}

async fn update(
    State(state): State<Shared>,
    Path((worker, round)): Path<(String, u64)>,
    body: Bytes,
) -> Response {
    let mut c = state.lock().unwrap();
    if !c.workers.contains_key(&worker) {
        return (StatusCode::NOT_FOUND, "unknown worker").into_response();
    }
    if round != c.round || c.finished {
        return (
            StatusCode::CONFLICT,
            format!("current round is {}", c.round),
        )
            .into_response();
    }
    if c.updates.contains_key(&worker) {
        return (StatusCode::CONFLICT, "already submitted for this round").into_response();
    }
    let delta = match decode_update(&body, c.params.len()) {
        Ok(delta) => delta,
        Err(e) => {
            println!("round {round}: rejected update from {worker}: {e}");
            return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
        }
    };
    println!(
        "round {round}: update from {worker}, ‖Δ‖ = {:.4}",
        l2(&delta)
    );
    c.updates.insert(worker, delta);
    if c.updates.len() >= c.min_workers {
        c.close_round();
    }
    StatusCode::ACCEPTED.into_response()
}

#[derive(Serialize)]
struct Status {
    run_id: String,
    round: u64,
    rounds: u64,
    workers: usize,
    pending_updates: usize,
    val_loss: f32,
    finished: bool,
}

async fn status(State(state): State<Shared>) -> Json<Status> {
    let c = state.lock().unwrap();
    Json(Status {
        run_id: c.run.run_id.clone(),
        round: c.round,
        rounds: c.rounds,
        workers: c.workers.len(),
        pending_updates: c.updates.len(),
        val_loss: c.val_loss,
        finished: c.finished,
    })
}

async fn close_stale_rounds(state: Shared) {
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    loop {
        tick.tick().await;
        let mut c = state.lock().unwrap();
        if !c.finished && !c.updates.is_empty() && c.round_started.elapsed() >= c.round_timeout {
            println!("round {}: timed out waiting for more workers", c.round);
            c.close_round();
        }
    }
}

impl Coordinator {
    fn close_round(&mut self) {
        let updates = std::mem::take(&mut self.updates);
        let from: Vec<&str> = updates.keys().map(String::as_str).collect();
        let deltas: Vec<Vec<f32>> = updates.values().cloned().collect();
        let pseudo_grad = robust_mean(&deltas, self.trim_fraction);
        self.outer.step(&mut self.params, &pseudo_grad);
        self.val_loss = self.model.loss(&self.params, &self.val);
        self.weights = Bytes::from(encode_f32s(&self.params));
        println!(
            "round {} closed with {} update(s) from {}: val loss {:.4}",
            self.round,
            deltas.len(),
            from.join(", "),
            self.val_loss
        );
        self.round += 1;
        self.round_started = Instant::now();
        if self.round >= self.rounds {
            self.finished = true;
            println!("run finished after {} rounds", self.rounds);
            if let Some(path) = &self.out {
                match std::fs::write(path, &self.weights) {
                    Ok(()) => println!("final weights written to {}", path.display()),
                    Err(e) => println!("could not write final weights to {}: {e}", path.display()),
                }
            }
        }
    }
}

/// Shuffles the corpus in fixed-size chunks before splitting off validation
/// data, so every shard samples the whole corpus the way a pre-shuffled
/// pretraining dataset would. Contiguous shards of unshuffled text are
/// non-IID, and the averaged model then chases whichever files it saw last.
fn split_corpus(corpus: &[u8]) -> (Vec<u8>, Vec<u8>) {
    const CHUNK: usize = 4 << 10;
    let mut chunks: Vec<&[u8]> = corpus.chunks(CHUNK).collect();
    let mut rng = dti_model::Rng::new(0x5EED);
    for i in (1..chunks.len()).rev() {
        chunks.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let val_chunks = (corpus.len() / 10).clamp(2, VAL_BYTES).div_ceil(CHUNK);
    let (val, train) = chunks.split_at(val_chunks.min(chunks.len() - 1));
    (train.concat(), val.concat())
}

fn splitmix(x: u64) -> u64 {
    dti_model::Rng::new(x).next_u64()
}
