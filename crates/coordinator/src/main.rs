//! Coordinates a DiLoCo training run: leases slices of training data to
//! workers, collects compressed pseudo-gradients from them, combines them
//! robustly and applies the outer optimiser step.
//!
//! Workers download their slices themselves and train on as much data as
//! their volunteers' budgets allow; what is trained on, how updates are
//! filtered and combined, and the optimiser all live here.
//!
//! This is a single trusted coordinator. Splitting it across independent
//! aggregators that agree on each round's result is future work.

mod aggregate;
mod data;

use aggregate::{Nesterov, l2, robust_mean};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use data::{Ledger, Plan, corpus_slices};
use dti_model::Model;
use dti_protocol::{
    JoinResponse, ModelSpec, RunInfo, SliceRequest, Task, TrainConfig, UpdateInfo, decode_update,
    encode_f32s, update_len,
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
    /// Training text served by the coordinator. The model is byte-level, so any file works.
    #[arg(long, required_unless_present = "dataset", conflicts_with = "dataset")]
    corpus: Option<PathBuf>,
    /// A dataset plan from `data/plan.py`; workers download its slices themselves.
    #[arg(long)]
    dataset: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    /// Close a round as soon as this many workers have sent updates.
    #[arg(long, default_value_t = 2)]
    min_workers: usize,
    /// Close a round after this long if at least one update has arrived.
    #[arg(long, default_value_t = 120)]
    round_timeout_secs: u64,
    /// Stop after this many rounds; by default the run ends when the data does.
    #[arg(long)]
    rounds: Option<u64>,
    /// Passes over the training data.
    #[arg(long, default_value_t = 1)]
    epochs: u32,
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
    /// Size of each slice of `--corpus`.
    #[arg(long, default_value_t = 64 << 10)]
    slice_bytes: u64,
    /// Put a worker's unfinished slices back in the pool after this long
    /// without hearing from it. Must be longer than a round takes.
    #[arg(long, default_value_t = 1800)]
    lease_secs: u64,
    /// Write the final weights (raw little-endian f32) here when the run ends.
    #[arg(long)]
    out: Option<PathBuf>,
}

struct Coordinator {
    run: RunInfo,
    model: Box<dyn Model>,
    /// Text served by `GET /v1/data`; empty for a dataset plan.
    train: Arc<[u8]>,
    val: Vec<u8>,
    ledger: Ledger,
    min_workers: usize,
    rounds: Option<u64>,
    trim_fraction: f32,
    round_timeout: Duration,
    lease: Duration,
    out: Option<PathBuf>,

    params: Vec<f32>,
    outer: Nesterov,
    /// `params` encoded for `GET /v1/weights`, rebuilt once per round.
    weights: Bytes,
    round: u64,
    round_started: Instant,
    /// Worker id → when it last contacted us.
    workers: HashMap<String, Instant>,
    updates: BTreeMap<String, Vec<f32>>,
    /// Bytes of data in slices that are done.
    trained_bytes: u64,
    val_loss: f32,
    finished: bool,
}

type Shared = Arc<Mutex<Coordinator>>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let (train, val, slices, source) = match (&cli.corpus, &cli.dataset) {
        (Some(corpus), _) => {
            let text = std::fs::read(corpus)?;
            anyhow::ensure!(text.len() >= 1024, "corpus is too small to train on");
            let (train, val) = split_corpus(&text);
            let slices = corpus_slices(train.len() as u64, cli.slice_bytes.max(2));
            (train, val, slices, corpus.display().to_string())
        }
        (None, Some(path)) => {
            let (plan, val) = Plan::load(path)?;
            (Vec::new(), val, plan.slices, plan.source)
        }
        (None, None) => unreachable!("clap requires --corpus or --dataset"),
    };
    let val = val[..val.len().min(VAL_BYTES)].to_vec();
    let total_bytes: u64 = slices.iter().map(|s| s.bytes).sum::<u64>() * u64::from(cli.epochs);
    let ledger = Ledger::new(slices, cli.epochs.max(1));

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
            data: source,
        },
        val_loss: model.loss(&params, &val),
        outer: Nesterov::new(params.len(), cli.outer_lr, cli.outer_momentum),
        weights: Bytes::from(encode_f32s(&params)),
        params,
        model,
        train: train.into(),
        val,
        ledger,
        min_workers: cli.min_workers.max(1),
        rounds: cli.rounds,
        trim_fraction: cli.trim_fraction,
        round_timeout: Duration::from_secs(cli.round_timeout_secs),
        lease: Duration::from_secs(cli.lease_secs),
        out: cli.out,
        round: 0,
        round_started: Instant::now(),
        workers: HashMap::new(),
        updates: BTreeMap::new(),
        trained_bytes: 0,
        finished: false,
    };
    coord.finished = coord.rounds == Some(0);

    println!(
        "{}: model {} ({} params), data {}: {} slices, {} over {} epoch(s), {} val bytes, {} inner steps per round",
        coord.run.run_id,
        coord.run.model,
        coord.run.num_params,
        coord.run.data,
        coord.ledger.len(),
        bytes(total_bytes),
        cli.epochs.max(1),
        coord.val.len(),
        coord.run.train.inner_steps,
    );
    println!("round 0: val loss {:.4}", coord.val_loss);

    let max_body = update_len(coord.params.len()) + 1024;
    let state: Shared = Arc::new(Mutex::new(coord));
    tokio::spawn(housekeeping(state.clone()));

    let app = Router::new()
        .route("/v1/join", post(join))
        .route("/v1/task/{worker}", get(task))
        .route("/v1/slice/{worker}", post(slice))
        .route("/v1/done/{worker}", post(done))
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
    let worker_id = format!("w{}", c.workers.len() + 1);
    c.workers.insert(worker_id.clone(), Instant::now());
    println!("{worker_id} joined ({} workers)", c.workers.len());
    Json(JoinResponse {
        worker_id,
        run: c.run.clone(),
    })
}

async fn task(State(state): State<Shared>, Path(worker): Path<String>) -> Response {
    let mut c = state.lock().unwrap();
    if !c.seen(&worker) {
        return (StatusCode::NOT_FOUND, "unknown worker").into_response();
    }
    if c.finished {
        return (StatusCode::GONE, "run finished").into_response();
    }
    Json(Task { round: c.round }).into_response()
}

/// Leases the worker its next slice, or 204 if none fits in its budget.
async fn slice(
    State(state): State<Shared>,
    Path(worker): Path<String>,
    Json(req): Json<SliceRequest>,
) -> Response {
    let mut c = state.lock().unwrap();
    if !c.seen(&worker) {
        return (StatusCode::NOT_FOUND, "unknown worker").into_response();
    }
    if c.finished {
        return StatusCode::NO_CONTENT.into_response();
    }
    match c.ledger.lease(&worker, req.max_bytes) {
        Some(slice) => Json(slice).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}

/// The worker has trained on all of these slices and its update was accepted.
async fn done(
    State(state): State<Shared>,
    Path(worker): Path<String>,
    Json(ids): Json<Vec<u64>>,
) -> Response {
    let mut c = state.lock().unwrap();
    if !c.seen(&worker) {
        return (StatusCode::NOT_FOUND, "unknown worker").into_response();
    }
    for id in ids {
        if let Some(bytes) = c.ledger.done(&worker, id) {
            c.trained_bytes += bytes;
        }
    }
    c.finish_if_data_done();
    StatusCode::NO_CONTENT.into_response()
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
    Query(info): Query<UpdateInfo>,
    body: Bytes,
) -> Response {
    let mut c = state.lock().unwrap();
    if !c.seen(&worker) {
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
    if info.tokens == 0 {
        return (StatusCode::BAD_REQUEST, "update trained on no tokens").into_response();
    }
    let delta = match decode_update(&body, c.params.len()) {
        Ok(delta) => delta,
        Err(e) => {
            println!("round {round}: rejected update from {worker}: {e}");
            return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
        }
    };
    println!(
        "round {round}: update from {worker}, {} tokens, ‖Δ‖ = {:.4}",
        info.tokens,
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
    workers: usize,
    pending_updates: usize,
    slices: usize,
    slices_leased: usize,
    slices_done: usize,
    trained_bytes: u64,
    val_loss: f32,
    finished: bool,
}

async fn status(State(state): State<Shared>) -> Json<Status> {
    let c = state.lock().unwrap();
    Json(Status {
        run_id: c.run.run_id.clone(),
        round: c.round,
        workers: c.workers.len(),
        pending_updates: c.updates.len(),
        slices: c.ledger.len(),
        slices_leased: c.ledger.leased_count(),
        slices_done: c.ledger.done_count(),
        trained_bytes: c.trained_bytes,
        val_loss: c.val_loss,
        finished: c.finished,
    })
}

/// Closes rounds that waited too long and takes slices back from workers
/// that went quiet.
async fn housekeeping(state: Shared) {
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    loop {
        tick.tick().await;
        let mut c = state.lock().unwrap();
        if !c.finished && !c.updates.is_empty() && c.round_started.elapsed() >= c.round_timeout {
            println!("round {}: timed out waiting for more workers", c.round);
            c.close_round();
        }
        let quiet: Vec<String> = c
            .workers
            .iter()
            .filter(|(_, seen)| seen.elapsed() >= c.lease)
            .map(|(w, _)| w.clone())
            .collect();
        for worker in quiet {
            let released = c.ledger.release(&worker);
            if released > 0 {
                println!("{worker} went quiet: {released} slice(s) back in the pool");
            }
        }
    }
}

impl Coordinator {
    /// Notes that `worker` is alive; false if it never joined.
    fn seen(&mut self, worker: &str) -> bool {
        match self.workers.get_mut(worker) {
            Some(seen) => {
                *seen = Instant::now();
                true
            }
            None => false,
        }
    }

    fn close_round(&mut self) {
        let updates = std::mem::take(&mut self.updates);
        let from: Vec<&str> = updates.keys().map(String::as_str).collect();
        let deltas: Vec<Vec<f32>> = updates.values().cloned().collect();
        let pseudo_grad = robust_mean(&deltas, self.trim_fraction);
        self.outer.step(&mut self.params, &pseudo_grad);
        self.val_loss = self.model.loss(&self.params, &self.val);
        self.weights = Bytes::from(encode_f32s(&self.params));
        println!(
            "round {} closed with {} update(s) from {}: val loss {:.4}, {}/{} slices done ({})",
            self.round,
            deltas.len(),
            from.join(", "),
            self.val_loss,
            self.ledger.done_count(),
            self.ledger.len(),
            bytes(self.trained_bytes),
        );
        self.round += 1;
        self.round_started = Instant::now();
        if self.rounds.is_some_and(|r| self.round >= r) {
            self.finish(&format!("after {} rounds", self.round));
        } else {
            self.finish_if_data_done();
        }
    }

    /// Ends the run once every slice is trained on and merged.
    fn finish_if_data_done(&mut self) {
        if !self.finished && self.updates.is_empty() && self.ledger.finished() {
            self.finish(&format!(
                "with all data trained on, after {} rounds",
                self.round
            ));
        }
    }

    fn finish(&mut self, why: &str) {
        self.finished = true;
        println!("run finished {why}");
        if let Some(path) = &self.out {
            match std::fs::write(path, &self.weights) {
                Ok(()) => println!("final weights written to {}", path.display()),
                Err(e) => println!("could not write final weights to {}: {e}", path.display()),
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

fn bytes(n: u64) -> String {
    match n {
        n if n >= 1 << 30 => format!("{:.2} GiB", n as f64 / (1u64 << 30) as f64),
        n if n >= 1 << 20 => format!("{:.1} MiB", n as f64 / (1u64 << 20) as f64),
        n if n >= 1 << 10 => format!("{:.1} KiB", n as f64 / 1024.0),
        n => format!("{n} B"),
    }
}
