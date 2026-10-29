//! Runs a `dti-worker` component with a deliberately small set of capabilities:
//!
//! * outbound HTTP(S) to exactly one origin, the coordinator
//! * no filesystem, environment variables or raw sockets
//! * a hard cap on linear memory
//! * `wasi:nn`, only if the config lists TorchScript graphs, and then only those
//!
//! Network and memory figures shown on screen are measured here, not reported
//! by the guest; training progress comes from the worker via `dti:runner/telemetry`.

mod config;
mod egress;
#[cfg(feature = "pytorch")]
mod nn;

use clap::Parser;
use config::{Cli, Config, Target};
use egress::{Egress, NetStats, Origin};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;
use wasmtime::component::{Component, HasSelf, Linker, ResourceTable};
use wasmtime::error::Context;
use wasmtime::{Engine, ResourceLimiter, Store, bail};
use wasmtime_wasi::p2::bindings::sync::Command;
use wasmtime_wasi::{I32Exit, WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpCtxView, WasiHttpView};

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

    println!("dti-runner {}", env!("CARGO_PKG_VERSION"));
    println!("  wasm         {}", config.wasm.display());
    println!("  sha256       {sha256}");
    println!(
        "  coordinator  {}  (only allowed destination)",
        config.coordinator
    );
    println!("  memory cap   {} MiB", config.max_memory_mb);
    println!("  filesystem   none");
    println!("  env vars     none");
    println!("  raw sockets  none");
    if config.graphs.is_empty() {
        println!("  wasi:nn      none");
    }
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
    if !config.graphs.is_empty() {
        println!(
            "  warning: wasi:nn tensors and PyTorch's own memory do not count toward the memory cap"
        );
    }

    let engine = Engine::default();
    let component = Component::new(&engine, &wasm).context("loading worker component")?;
    let imports_nn = component
        .component_type()
        .imports(&engine)
        .any(|(name, _)| name.starts_with("wasi:nn/"));
    if imports_nn && config.graphs.is_empty() {
        bail!("the worker imports wasi:nn but the config lists no [graphs] for it to run");
    }
    if !imports_nn && !config.graphs.is_empty() {
        println!("  warning: [graphs] are configured but the worker does not import wasi:nn");
    }
    println!();

    let mut linker = Linker::<State>::new(&engine);
    wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
    wasmtime_wasi_http::p2::add_only_http_to_linker_sync(&mut linker)?;
    telemetry::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)?;
    #[cfg(feature = "pytorch")]
    let nn = nn::load(&config.graphs)?;
    #[cfg(feature = "pytorch")]
    if !config.graphs.is_empty() {
        wasmtime_wasi_nn::wit::add_to_linker(&mut linker, |state: &mut State| {
            wasmtime_wasi_nn::wit::WasiNnView::new(&mut state.table, &mut state.nn)
        })?;
    }

    let net = Arc::new(NetStats::default());
    let state = State {
        // No preopens, no env, no stdin; raw sockets are denied by default.
        wasi: WasiCtxBuilder::new()
            .args(&["dti-worker", &config.coordinator])
            .inherit_stdout()
            .inherit_stderr()
            .build(),
        http: WasiHttpCtx::new(),
        #[cfg(feature = "pytorch")]
        nn,
        table: ResourceTable::new(),
        egress: Egress {
            allowed: origin,
            stats: net.clone(),
        },
        memory: MemoryCap {
            max: usize::try_from(config.max_memory_mb)?.saturating_mul(MIB),
            current: 0,
            peak: 0,
        },
        started: Instant::now(),
        net: net.clone(),
    };
    let mut store = Store::new(&engine, state);
    store.limiter(|state| &mut state.memory);

    let result = Command::instantiate(&mut store, &component, &linker)
        .context("instantiating worker (does it import something the runner does not grant?)")
        .and_then(|command| command.wasi_cli_run().call_run(&mut store));

    let state = store.data();
    println!();
    println!(
        "[runner] {:.0?} elapsed, {} requests, {} denied, sent {}, received {}, peak memory {}",
        state.started.elapsed(),
        net.requests.load(Relaxed),
        net.denied.load(Relaxed),
        bytes(net.bytes_up.load(Relaxed)),
        bytes(net.bytes_down.load(Relaxed)),
        bytes(state.memory.peak as u64),
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
    #[cfg(feature = "pytorch")]
    nn: wasmtime_wasi_nn::wit::WasiNnCtx,
    table: ResourceTable,
    egress: Egress,
    memory: MemoryCap,
    started: Instant,
    net: Arc<NetStats>,
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
            "net ↑{} ↓{}  mem {}",
            bytes(self.net.bytes_up.load(Relaxed)),
            bytes(self.net.bytes_down.load(Relaxed)),
            bytes(self.memory.current as u64),
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

fn bytes(n: u64) -> String {
    match n {
        n if n >= 1 << 30 => format!("{:.1} GiB", n as f64 / (1u64 << 30) as f64),
        n if n >= 1 << 20 => format!("{:.1} MiB", n as f64 / (1u64 << 20) as f64),
        n if n >= 1 << 10 => format!("{:.1} KiB", n as f64 / 1024.0),
        n => format!("{n} B"),
    }
}
