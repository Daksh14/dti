use clap::Parser;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use wasmtime::bail;
use wasmtime::error::Context;

const DEFAULT_MAX_MEMORY_MB: u64 = 1024;
const DEFAULT_DATA_GB: f64 = 1.0;
/// Hugging Face, and the CDN its `resolve` URLs redirect to.
const DEFAULT_DATA_ORIGINS: [&str; 2] = ["https://huggingface.co", "https://*.hf.co"];

/// Run a dti training worker in a sandbox that can only reach its coordinator
/// and the data origins you allow, and can only train with your TorchScript graphs.
#[derive(Parser)]
#[command(version)]
pub struct Cli {
    /// The worker component to run. Build it yourself with `cargo build-worker`.
    pub wasm: PathBuf,

    /// TOML config file; command-line flags override it.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Coordinator URL.
    #[arg(long)]
    pub coordinator: Option<String>,

    /// How much training data to download, in GB. This sets how long you
    /// train: every downloaded byte is trained on once.
    #[arg(long)]
    pub data_gb: Option<f64>,

    /// Origin the worker may download training data from, e.g.
    /// `https://huggingface.co` or `https://*.hf.co`. Repeat for several;
    /// replaces the config's list.
    #[arg(long = "data-origin")]
    pub data_origins: Vec<String>,

    /// Cap on the worker's linear memory, in MiB.
    #[arg(long)]
    pub max_memory_mb: Option<u64>,

    /// CPU threads libtorch may use (default: libtorch's choice).
    #[arg(long)]
    pub torch_threads: Option<usize>,

    /// Refuse to start unless the wasm file has this SHA-256 (hex).
    #[arg(long)]
    pub expect_sha256: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct FileConfig {
    coordinator: Option<String>,
    data_gb: Option<f64>,
    data_origins: Option<Vec<String>>,
    max_memory_mb: Option<u64>,
    torch_threads: Option<usize>,
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
    pub sha256: String,
    pub target: Target,
}

pub struct Config {
    pub wasm: PathBuf,
    pub coordinator: String,
    /// Most bytes the worker may download from `data_origins`.
    pub data_budget: u64,
    pub data_origins: Vec<String>,
    pub max_memory_mb: u64,
    pub torch_threads: Option<usize>,
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
        let data_gb = cli.data_gb.or(file.data_gb).unwrap_or(DEFAULT_DATA_GB);
        if !(data_gb.is_finite() && data_gb > 0.0) {
            bail!("data-gb must be a positive number of gigabytes");
        }
        let data_origins = if cli.data_origins.is_empty() {
            file.data_origins
                .unwrap_or_else(|| DEFAULT_DATA_ORIGINS.map(String::from).to_vec())
        } else {
            cli.data_origins
        };
        if cli.torch_threads.or(file.torch_threads) == Some(0) {
            bail!("torch-threads must be greater than zero");
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
            data_budget: (data_gb * 1e9) as u64,
            data_origins,
            max_memory_mb,
            torch_threads: cli.torch_threads.or(file.torch_threads),
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
