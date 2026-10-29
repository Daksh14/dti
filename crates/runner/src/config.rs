use clap::Parser;
use serde::Deserialize;
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
}

pub struct Config {
    pub wasm: PathBuf,
    pub coordinator: String,
    pub max_memory_mb: u64,
    /// Accepted so config files can carry it, but nothing enforces it until a
    /// `wasi:webgpu` backend is wired up.
    pub gpu_memory_mb: Option<u64>,
    pub expected_sha256: Option<String>,
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
        Ok(Self {
            wasm: cli.wasm,
            coordinator,
            max_memory_mb,
            gpu_memory_mb: file.gpu_memory_mb,
            expected_sha256: cli
                .expect_sha256
                .or(file.expected_sha256)
                .map(|h| h.trim().to_ascii_lowercase()),
        })
    }
}

fn read_file(path: &Path) -> wasmtime::Result<FileConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
}
