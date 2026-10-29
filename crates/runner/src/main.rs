//! Runs a `dti-worker` component with a deliberately small set of capabilities:
//!
//! * outbound HTTP(S) to the coordinator, and to the volunteer's data origins
//!   for at most their data budget
//! * `wasi:nn` with libtorch, for the TorchScript graphs the config lists only
//! * no filesystem, environment variables or raw sockets
//! * a hard cap on linear memory
//!
//! Network, memory and libtorch figures shown on screen are measured here, not
//! reported by the guest; training progress comes from the worker via
//! `dti:runner/telemetry`.

mod config;
mod egress;
mod nn;

use clap::Parser;
use config::{Cli, Config, Target};
use egress::{Egress, NetStats, Origin};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};
use wasmtime::component::{Component, HasSelf, Linker, ResourceTable};
use wasmtime::error::Context;
use wasmtime::{Engine, ResourceLimiter, Store, bail};
use wasmtime_wasi::p2::bindings::sync::Command;
use wasmtime_wasi::{I32Exit, WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpCtxView, WasiHttpView};
use wasmtime_wasi_nn::wit::{WasiNnCtx, WasiNnView};

wasmtime::component::bindgen!({
    path: "../../wit",
    world: "worker",
});

use dti::runner::telemetry;

const MIB: usize = 1024 * 1024;

fn main() -> wasmtime::Result<()> {
    let config = Config::load(Cli::parse())?;
    let wasm = std::fs::read(&config.wasm)
        .with_context(|| format!("reading {}", config.wasm.display()))?;
    let sha256 = hex(&Sha256::digest(&wasm));
    let origin = Origin::parse(&config.coordinator)?;
    let data_origins = config
        .data_origins
        .iter()
        .map(|o| Origin::parse(o).with_context(|| format!("data origin {o:?}")))
        .collect::<wasmtime::Result<Vec<_>>>()?;

    println!("dti-runner {}", env!("CARGO_PKG_VERSION"));
    println!("  wasm         {}", config.wasm.display());
    println!("  sha256       {sha256}");
    println!("  coordinator  {}", config.coordinator);
    println!(
        "  data         up to {} from {}",
        gb(config.data_budget),
        if config.data_origins.is_empty() {
            "nowhere but the coordinator".to_string()
        } else {
            config.data_origins.join(", ")
        }
    );
    println!("  network      only the coordinator and the data origins");
    println!("  memory cap   {} MiB", config.max_memory_mb);
    println!("  filesystem   none");
    println!("  env vars     none");
    println!("  raw sockets  none");
    for graph in &config.graphs {
        println!(
            "  wasi:nn      {:?} from {}, PyTorch on {}",
            graph.name,
            graph.path.display(),
            match graph.target {
                Target::Cpu => "cpu",
                Target::Gpu => "gpu",
            }
        );
    }
    if let Some(threads) = config.torch_threads {
        tch::set_num_threads(threads as i32);
    }
    println!("  torch        {} CPU threads", tch::get_num_threads());
    if config.graphs.iter().any(|g| g.target == Target::Gpu) {
        println!("  gpu          cuda:0, for wasi:nn graphs only");
    } else {
        println!("  gpu          none");
    }
    if let Some(expected) = &config.expected_sha256 {
        if *expected != sha256 {
            bail!("wasm SHA-256 mismatch: expected {expected}, got {sha256}");
        }
        println!("  sha256 matches the expected hash");
    }
    if !origin.is_https() {
        println!("  warning: coordinator is plain http, traffic is not encrypted");
    }
    if config.gpu_memory_mb.is_some_and(|mb| mb > 0) {
        println!("  warning: gpu-memory-mb is not enforced yet");
    }
    println!(
        "  warning: wasi:nn tensors and PyTorch's own memory do not count toward the memory cap"
    );
    if config.graphs.is_empty() {
        bail!(
            "the config lists no [graphs]; workers train only through wasi:nn, \
             so export one with graphs/export.py and add it"
        );
    }

    let engine = Engine::default();
    let component = Component::new(&engine, &wasm).context("loading worker component")?;
    println!();

    let mut linker = Linker::<State>::new(&engine);
    wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
    wasmtime_wasi_http::p2::add_only_http_to_linker_sync(&mut linker)?;
    telemetry::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)?;
    wasmtime_wasi_nn::wit::add_to_linker(&mut linker, |state: &mut State| {
        WasiNnView::new(&mut state.table, &mut state.nn)
    })?;
    let torch_nanos = Arc::new(AtomicU64::new(0));
    let nn = nn::load(&config.graphs, torch_nanos.clone())?;

    let net = Arc::new(NetStats::default());
    let budget = config.data_budget.to_string();
    let state = State {
        // No preopens, no env, no stdin; raw sockets are denied by default.
        wasi: WasiCtxBuilder::new()
            .args(&["dti-worker", &config.coordinator, &budget])
            .inherit_stdout()
            .inherit_stderr()
            .build(),
        http: WasiHttpCtx::new(),
        nn,
        table: ResourceTable::new(),
        egress: Egress {
            coordinator: origin,
            data: data_origins,
            data_budget: config.data_budget,
            stats: net.clone(),
        },
        memory: MemoryCap {
            max: usize::try_from(config.max_memory_mb)?.saturating_mul(MIB),
            current: 0,
            peak: 0,
        },
        started: Instant::now(),
        net: net.clone(),
        data_budget: config.data_budget,
        torch_nanos: torch_nanos.clone(),
    };
    let mut store = Store::new(&engine, state);
    store.limiter(|state| &mut state.memory);

    let result = Command::instantiate(&mut store, &component, &linker)
        .context("instantiating worker (does it import something the runner does not grant?)")
        .and_then(|command| command.wasi_cli_run().call_run(&mut store));

    let state = store.data();
    println!();
    println!(
        "[runner] {:.0?} elapsed, {} requests, {} denied, sent {}, received {} ({} of {} data), \
         peak memory {}, {:.1?} in libtorch",
        state.started.elapsed(),
        net.requests.load(Relaxed),
        net.denied.load(Relaxed),
        bytes(net.bytes_up.load(Relaxed)),
        bytes(net.bytes_down.load(Relaxed)),
        gb(net.data_down.load(Relaxed)),
        gb(config.data_budget),
        bytes(state.memory.peak as u64),
        Duration::from_nanos(torch_nanos.load(Relaxed)),
    );

    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(())) => std::process::exit(1),
        Err(e) => match e.downcast_ref::<I32Exit>() {
            Some(exit) => std::process::exit(exit.0),
            None => Err(e),
        },
    }
}

struct State {
    wasi: WasiCtx,
    http: WasiHttpCtx,
    nn: WasiNnCtx,
    table: ResourceTable,
    egress: Egress,
    memory: MemoryCap,
    started: Instant,
    net: Arc<NetStats>,
    data_budget: u64,
    torch_nanos: Arc<AtomicU64>,
}

impl WasiView for State {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl WasiHttpView for State {
    fn http(&mut self) -> WasiHttpCtxView<'_> {
        WasiHttpCtxView {
            ctx: &mut self.http,
            table: &mut self.table,
            hooks: &mut self.egress,
        }
    }
}

impl State {
    /// Host-measured figures appended to each guest report.
    fn usage(&self) -> String {
        format!(
            "net ↑{} ↓{}  data {}/{}  mem {}  torch {:.1?}",
            bytes(self.net.bytes_up.load(Relaxed)),
            bytes(self.net.bytes_down.load(Relaxed)),
            gb(self.net.data_down.load(Relaxed)),
            gb(self.data_budget),
            bytes(self.memory.current as u64),
            Duration::from_nanos(self.torch_nanos.load(Relaxed)),
        )
    }
}

impl telemetry::Host for State {
    fn joined(&mut self, info: telemetry::RunInfo) {
        println!(
            "[run]    joined {} as {}: model {} with {} params",
            info.run_id, info.worker_id, info.model, info.params
        );
    }

    fn step(&mut self, m: telemetry::StepMetrics) {
        println!(
            "[train]  round {:>4}  step {:>5}/{:<5}  loss {:.4}   | {}",
            m.round,
            m.step,
            m.inner_steps,
            m.loss,
            self.usage()
        );
    }

    fn round_finished(&mut self, round: u64, update_bytes: u64) {
        println!(
            "[train]  round {round:>4}  update sent ({})   | {}",
            bytes(update_bytes),
            self.usage()
        );
    }

    fn log(&mut self, message: String) {
        println!("[worker] {message}");
    }
}

/// Caps the total linear memory of the worker and tracks its size for display.
struct MemoryCap {
    max: usize,
    current: usize,
    peak: usize,
}

impl ResourceLimiter for MemoryCap {
    fn memory_growing(
        &mut self,
        current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        let total = self.current - current + desired;
        if total > self.max {
            bail!(
                "worker tried to grow memory to {}, over the {} cap",
                bytes(total as u64),
                bytes(self.max as u64)
            );
        }
        self.current = total;
        self.peak = self.peak.max(total);
        Ok(true)
    }

    fn table_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        Ok(desired <= 1_000_000)
    }
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decimal gigabytes, the unit of `data-gb`.
fn gb(n: u64) -> String {
    format!("{:.3} GB", n as f64 / 1e9)
}

fn bytes(n: u64) -> String {
    match n {
        n if n >= 1 << 30 => format!("{:.1} GiB", n as f64 / (1u64 << 30) as f64),
        n if n >= 1 << 20 => format!("{:.1} MiB", n as f64 / (1u64 << 20) as f64),
        n if n >= 1 << 10 => format!("{:.1} KiB", n as f64 / 1024.0),
        n => format!("{n} B"),
    }
}
