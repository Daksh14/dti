use clap::Parser;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use wasmtime::bail;
use wasmtime::error::Context;

const DEFAULT_MAX_MEMORY_MB: u64 = 1024;

/// Run a dti training worker in a sandbox that can only reach its coordinator.
#[derive(Parser)]
#[command(version)]
pub struct Cli {
    /// The worker component to run. Build it yourself with `cargo build-worker`.
    pub wasm: PathBuf,

    /// TOML config file; command-line flags override it.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Coordinator URL, the only destination the worker may reach.
    #[arg(long)]
    pub coordinator: Option<String>,

    /// Cap on the worker's linear memory, in MiB.
    #[arg(long)]
    pub max_memory_mb: Option<u64>,

    /// Refuse to start unless the wasm file has this SHA-256 (hex).
    #[arg(long)]
    pub expect_sha256: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct FileConfig {
    coordinator: Option<String>,
    max_memory_mb: Option<u64>,
    gpu_memory_mb: Option<u64>,
    expected_sha256: Option<String>,
    #[serde(default)]
    graphs: BTreeMap<String, GraphFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct GraphFile {
    path: PathBuf,
    sha256: String,
    #[serde(default)]
    target: Target,
}

/// Where the PyTorch backend runs a graph.
#[derive(Deserialize, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Target {
    #[default]
    Cpu,
    /// CUDA device 0; needs a CUDA build of libtorch.
    Gpu,
}

/// A TorchScript graph the worker may run through `wasi:nn`, by name.
///
/// The graph executes as native code in the runner, outside the sandbox,
/// which is why its hash must be pinned.
pub struct Graph {
    pub name: String,
    pub path: PathBuf,
    #[cfg_attr(
        not(feature = "pytorch"),
        expect(dead_code, reason = "checked when loading")
    )]
    pub sha256: String,
    pub target: Target,
}

pub struct Config {
    pub wasm: PathBuf,
    pub coordinator: String,
    pub max_memory_mb: u64,
    /// Accepted so config files can carry it, but not enforced: `wasi:nn`
    /// graphs on the GPU are not limited yet.
    pub gpu_memory_mb: Option<u64>,
    pub expected_sha256: Option<String>,
    pub graphs: Vec<Graph>,
}

impl Config {
    pub fn load(cli: Cli) -> wasmtime::Result<Self> {
        let file = match &cli.config {
            Some(path) => read_file(path)?,
            None => FileConfig::default(),
        };
        let Some(coordinator) = cli.coordinator.or(file.coordinator) else {
            bail!("no coordinator configured: pass --coordinator or set `coordinator` in --config");
        };
        let max_memory_mb = cli
            .max_memory_mb
            .or(file.max_memory_mb)
            .unwrap_or(DEFAULT_MAX_MEMORY_MB);
        if max_memory_mb == 0 {
            bail!("max-memory-mb must be greater than zero");
        }
        if !file.graphs.is_empty() && !cfg!(feature = "pytorch") {
            bail!(
                "the config lists [graphs] but this runner was built without PyTorch; \
                 rebuild it with `cargo build --release --features pytorch`"
            );
        }
        // Graph paths are relative to the config file.
        let base = cli
            .config
            .as_deref()
            .and_then(Path::parent)
            .unwrap_or(Path::new(""));
        let graphs = file
            .graphs
            .into_iter()
            .map(|(name, g)| Graph {
                name,
                path: base.join(g.path),
                sha256: g.sha256.trim().to_ascii_lowercase(),
                target: g.target,
            })
            .collect();
        Ok(Self {
            wasm: cli.wasm,
            coordinator,
            max_memory_mb,
            gpu_memory_mb: file.gpu_memory_mb,
            expected_sha256: cli
                .expect_sha256
                .or(file.expected_sha256)
                .map(|h| h.trim().to_ascii_lowercase()),
            graphs,
        })
    }
}

fn read_file(path: &Path) -> wasmtime::Result<FileConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
}
