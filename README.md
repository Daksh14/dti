# dti

Distributed LLM training on volunteers' machines. Each volunteer runs a
training worker compiled to WebAssembly inside `dti-runner`, a sandbox that
lets it reach only the coordinator. The coordinator merges the workers'
updates DiLoCo-style: workers train locally for many steps and send one
compressed update per round, so it works over home internet connections.

## Layout

| Path | Runs | Role |
|---|---|---|
| `crates/runner` | volunteer's machine (native) | Embeds wasmtime. Grants the worker HTTP to the coordinator and nothing else, caps its memory, and shows the wasm hash, network use and training progress. |
| `crates/worker` | inside the runner (`wasm32-wasip2`) | Each round: pull global weights, run local AdamW steps on a data shard, push the int8-compressed weight difference. |
| `crates/coordinator` | server | Hands out shards, combines updates robustly (norm clipping + trimmed mean), applies the outer Nesterov step, tracks validation loss. |
| `crates/protocol` | both | Wire types and the int8 update codec. |
| `crates/model` | both | Model and AdamW. Currently a byte-level bigram placeholder. |
| `wit/dti.wit` | | Telemetry interface between worker and runner. |

## Quick start

```sh
rustup target add wasm32-wasip2
cargo build-worker      # -> target/wasm32-wasip2/release/dti-worker.wasm
cargo build --release

# Terminal 1: any text file works as a corpus (the model is byte-level).
./target/release/dti-coordinator --corpus corpus.txt

# Terminals 2 and 3: one runner per volunteer.
./target/release/dti-runner target/wasm32-wasip2/release/dti-worker.wasm \
    --coordinator http://127.0.0.1:8080
```

The runner also reads a TOML config (`--config`, see `dti.example.toml`);
command-line flags override it.

## What the sandbox allows

| Capability | Worker gets |
|---|---|
| Outbound HTTP(S) | Only to the configured coordinator (same scheme, host and port). TLS is done by the host. |
| Raw TCP/UDP sockets, DNS | None |
| Filesystem | None (no preopened directories) |
| Environment variables, stdin | None |
| Memory | Capped by `--max-memory-mb`; exceeding it stops the worker |
| GPU | None yet |
| Telemetry | `dti:runner/telemetry`, for progress display only |

All of these are enforced by the runner, not by trusting the guest.

A sandbox alone cannot rule out crypto mining: mining is just arithmetic plus
a connection to a pool. What protects you is that you build the worker
yourself from source and check its hash, and that the worker can reach only
the coordinator.

## Verifying the worker

The runner prints the wasm's SHA-256 on startup. Pass `--expect-sha256`
(or set `expected-sha256` in the config) to refuse anything else. Getting
the same hash on every machine also needs a pinned toolchain and trimmed
paths, which are not set up yet.

## Coordinator API

| Endpoint | Purpose |
|---|---|
| `POST /v1/join` | Register; returns a worker id and the run config |
| `GET /v1/task/{worker}` | Current round, data shard and batch-sampling seed (410 when the run is over) |
| `GET /v1/weights/{round}` | Global weights as little-endian f32 |
| `GET /v1/data/{offset}/{len}` | A slice of the training corpus |
| `POST /v1/update/{worker}/{round}` | int8-compressed `global - local` weights |
| `GET /v1/status` | Round, workers, validation loss |

## Not done yet

- **GPU**: embed [wasi-gfx-runtime](https://github.com/wasi-gfx/wasi-gfx-runtime)
  for `wasi:webgpu` and enforce `gpu-memory-mb`.
- **Real model**: replace the bigram with a transformer running on the GPU.
- **Spot checks**: replay a sample of a worker's steps (batch sampling is
  already seeded per task) and reject updates that don't match within tolerance.
- **Reproducible worker builds**, so everyone gets the same hash.
- **Coordinator persistence and worker auth**: checkpoints, resume, rate limits.
- **Multiple coordinators** agreeing on each round's result, if no single
  operator should control the run.
