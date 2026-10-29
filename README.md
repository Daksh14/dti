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
| `wit/dti.wit` | | Telemetry interface between worker and runner, and the `worker` / `worker-nn` worlds. |
| `graphs/export.py` | volunteer's machine | Exports a model as a TorchScript graph for the runner's `wasi:nn` backend. |

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

## Training steps in PyTorch through `wasi:nn`

A worker built with `cargo build-worker-nn` targets the `worker-nn` world: it
also imports `wasi:nn` and, if the runner has a graph named after the model
(e.g. `byte-bigram`), hands each step's forward and backward pass to it.
Otherwise it logs why and trains in wasm as usual. The plain `cargo
build-worker` wasm does not import `wasi:nn` at all.

The graph is TorchScript that maps `params: f32[n]` and `tokens: u8[t]` to a
single tensor `[loss, grad...]`, the same numbers the Rust model computes
(Wasmtime's PyTorch backend returns one output tensor). The runner executes it
with libtorch, on the CPU or a CUDA GPU.

```sh
# libtorch 2.4 is what wasmtime-wasi-nn 50 (via tch 0.17) links against;
# torch 2.4 has wheels for Python 3.8 to 3.12.
python3.12 -m venv .venv && .venv/bin/pip install torch==2.4.0
.venv/bin/python graphs/export.py byte-bigram byte-bigram.pt   # prints the config entry

PATH=$PWD/.venv/bin:$PATH LIBTORCH_USE_PYTORCH=1 \
    cargo build --release -p dti-runner --features pytorch
cargo build-worker-nn

# Add the printed [graphs.byte-bigram] entry to dti.toml, then:
export DYLD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/torch/lib  # LD_LIBRARY_PATH on Linux
./target/release/dti-runner --config dti.toml target/wasm32-wasip2/release/dti-worker.wasm
```

To build against a standalone libtorch instead, set `LIBTORCH=/path/to/libtorch`
in place of `LIBTORCH_USE_PYTORCH`. With a recent Xcode, the libtorch 2.4
headers also need `CXXFLAGS=-Wno-invalid-specialization`.

- **GPU** means CUDA device 0 and a CUDA build of libtorch. Apple's MPS is not
  supported by the backend. The runner refuses to start a `gpu` graph if
  libtorch finds no CUDA device.
- **The graph is not sandboxed.** TorchScript runs as native code in the
  runner and has ops that touch the host, such as reading files. So the
  worker can only `load-by-name` the graphs in the config, `load` from
  bytes the worker sends always fails, and each graph's SHA-256 must be
  pinned. Only use graphs you exported yourself or whose hash you have checked.
- **Guest tensors are checked** before PyTorch sees them. The backend copies
  as many bytes as a tensor's shape says without checking its data length,
  so an unchecked tensor could make the runner read past a buffer.
- **The memory cap does not cover** tensors the worker hands to `wasi:nn` or
  libtorch's own allocations, and `gpu-memory-mb` is not enforced.
- **The hash depends on** the torch version and on where `graphs/export.py`
  lives, because TorchScript keeps the source path for error messages. As
  with the worker wasm, everyone gets the same hash only with a pinned setup.

## What the sandbox allows

| Capability | Worker gets |
|---|---|
| Outbound HTTP(S) | Only to the configured coordinator (same scheme, host and port). TLS is done by the host. |
| Raw TCP/UDP sockets, DNS | None |
| Filesystem | None (no preopened directories) |
| Environment variables, stdin | None |
| Memory | Capped by `--max-memory-mb`; exceeding it stops the worker |
| `wasi:nn` | Only with `[graphs]` in the config, and then only those graphs, by name |
| GPU | Only through `wasi:nn` graphs with `target = "gpu"` |
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

- **GPU limits**: enforce `gpu-memory-mb` for `wasi:nn` graphs, or embed
  [wasi-gfx-runtime](https://github.com/wasi-gfx/wasi-gfx-runtime) for
  `wasi:webgpu` as an option that stays inside the sandbox.
- **Real model**: replace the bigram with a transformer. With `wasi:nn`, the
  parameters go to the graph and the gradient comes back on every step, which
  costs too much for large models. A graph that keeps the parameters and
  optimizer state on the device would avoid that.
- **Spot checks**: replay a sample of a worker's steps (batch sampling is
  already seeded per task) and reject updates that don't match within tolerance.
- **Reproducible worker builds**, so everyone gets the same hash.
- **Coordinator persistence and worker auth**: checkpoints, resume, rate limits.
- **Multiple coordinators** agreeing on each round's result, if no single
  operator should control the run.
