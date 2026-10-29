# dti

Distributed LLM training on volunteers' machines. Each volunteer runs a
training worker compiled to WebAssembly inside `dti-runner`, a sandbox that
lets it reach only the coordinator and the data origins the volunteer allows,
and train only with libtorch through `wasi:nn`. Each volunteer picks how many
GB of training data to download; the worker trains on every byte of it once,
so that choice sets how much CPU or GPU time they give. The coordinator
decides what is trained on and merges the workers' updates DiLoCo-style:
workers train locally for many steps and send one compressed update per
round, so it works over home internet connections.

## Layout

| Path | Runs | Role |
|---|---|---|
| `crates/runner` | volunteer's machine (native) | Embeds wasmtime and libtorch. Grants the worker HTTP to the coordinator and the data origins (up to the data budget), `wasi:nn` for the configured graphs, and nothing else. Shows the wasm hash, network, data, memory and libtorch use. |
| `crates/worker` | inside the runner (`wasm32-wasip2`) | Leases slices of data, downloads and decodes them, and each round pulls global weights, runs local AdamW steps through `wasi:nn` and pushes the int8-compressed weight difference. |
| `crates/coordinator` | server | Leases data slices, filters and combines updates (norm clipping + trimmed mean), applies the outer Nesterov step, tracks validation loss. |
| `crates/protocol` | both | Wire types and the int8 update codec. |
| `crates/model` | both | Model and AdamW. Currently a byte-level bigram placeholder. |
| `wit/dti.wit` | | The `worker` world: telemetry plus `wasi:nn`. |
| `graphs/export.py` | volunteer's machine | Exports a model as a TorchScript graph for the runner. |
| `data/plan.py` | coordinator operator | Turns a Hugging Face parquet dataset into slices workers download directly. |

## Quick start

The runner links libtorch 2.4, which is what `wasmtime-wasi-nn` 50 (via
`tch` 0.17) needs. torch 2.4 has wheels for Python 3.8 to 3.12.

```sh
rustup target add wasm32-wasip2
python3.12 -m venv .venv && .venv/bin/pip install torch==2.4.0

cargo build-worker      # -> target/wasm32-wasip2/release/dti-worker.wasm
cargo build --release   # coordinator
PATH=$PWD/.venv/bin:$PATH LIBTORCH_USE_PYTORCH=1 cargo build-runner

# Export the model's graph; this prints the [graphs.byte-bigram] entry.
.venv/bin/python graphs/export.py byte-bigram byte-bigram.pt
cp dti.example.toml dti.toml   # set coordinator, data-gb and the graph entry

# Terminal 1: any text file works as a corpus (the model is byte-level).
./target/release/dti-coordinator --corpus corpus.txt --epochs 3

# Terminals 2 and 3: one runner per volunteer.
export DYLD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/torch/lib  # LD_LIBRARY_PATH on Linux
./target/release/dti-runner --config dti.toml target/wasm32-wasip2/release/dti-worker.wasm
```

To build against a standalone libtorch instead, set `LIBTORCH=/path/to/libtorch`
in place of `LIBTORCH_USE_PYTORCH`. With a recent Xcode, the libtorch 2.4
headers also need `CXXFLAGS=-Wno-invalid-specialization`. Command-line flags
override the config file.

## Training on Hugging Face data

```sh
.venv/bin/pip install 'pyarrow<18'   # numpy 1.x, to match torch 2.4
.venv/bin/python data/plan.py HuggingFaceFW/fineweb-edu 'sample/10BT/*.parquet' fineweb-edu.json \
    --filter 'int_score>=3'
./target/release/dti-coordinator --dataset fineweb-edu.json
```

Volunteers run the same runner with `data-gb` set to what they want to give.

- **Slices are parquet row groups.** The planner reads each file's footer
  (two range requests per file, never the text) and pins the dataset to a
  commit. For each row group it records the byte range of the text column
  chunk and how to decode it. FineWeb-Edu's row groups are 1,000 documents,
  about 3 MB each.
- **Workers download their slices themselves**, straight from Hugging Face
  with one range request each, and decode them inside wasm. The coordinator
  never serves the text, so it doesn't need the bandwidth.
- **The coordinator decides what is trained on.** Each `--filter` (on
  `int_score`, `language_score`, `token_count`, ...) is evaluated by the
  planner on that column, which is tiny next to the text. Each slice carries a
  bit mask of the rows workers keep, and slices with nothing left are dropped.
  The coordinator also shuffles slices across files and holds the first row
  group out for validation.
- **Each slice goes to one worker at a time.** A worker leases the next slice
  that fits in what is left of its budget, and reports it done once the update
  that used its last byte is accepted. If a worker goes quiet for
  `--lease-secs`, its unfinished slices go back to the pool. The run ends when
  every slice is done, or after `--rounds`.
- **A failed round is retried on the same text**, so data isn't skipped
  because of a dropped connection.

Text columns must be snappy-compressed or uncompressed, which covers FineWeb
and Hugging Face's own parquet conversions, and datasets must be public.

## Data budget and compute

`data-gb` (or `--data-gb`) is how much training data the volunteer downloads.
Every byte is trained on once, in order, so compute is proportional to it.
The runner shows downloaded data against the budget and the time spent in
libtorch, and `torch-threads` limits libtorch's CPU threads.

- **The worker plans within the budget.** It asks only for slices that fit
  what is left of it, and stops when none do.
- **The runner enforces the budget**, whatever the worker or the coordinator
  claims. Once successful responses from data origins add up to the budget,
  it cuts off the download in progress and refuses further data requests.
  Slices of a `--corpus` served by the coordinator count toward the worker's
  budget, but the runner's hard cap covers only the data origins.

## Training through `wasi:nn`

Workers train only through `wasi:nn`. The runner executes the graph named
after the model (e.g. `byte-bigram`) with libtorch, on the CPU or a CUDA GPU.
The graph is TorchScript that maps `params: f32[n]` and `tokens: u8[t]` to a
single tensor `[loss, grad...]`, the same numbers the Rust model computes,
because Wasmtime's PyTorch backend returns one output tensor.

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
| Outbound HTTP(S) | The coordinator, and the data origins (default `https://huggingface.co` and `https://*.hf.co`, its CDN) up to the data budget. Same scheme, host and port; `*.` matches subdomains. TLS is done by the host. |
| Raw TCP/UDP sockets, DNS | None |
| Filesystem | None (no preopened directories) |
| Environment variables, stdin | None |
| Memory | Capped by `--max-memory-mb`; exceeding it stops the worker |
| `wasi:nn` | Only the `[graphs]` in the config, by name |
| GPU | Only through `wasi:nn` graphs with `target = "gpu"` |
| Telemetry | `dti:runner/telemetry`, for progress display only |

All of these are enforced by the runner, not by trusting the guest.

A sandbox alone cannot rule out crypto mining: mining is just arithmetic plus
a connection to a pool. What protects you is that you build the worker
yourself from source and check its hash, and that the worker can reach only
the coordinator and the data origins.

## Verifying the worker

The runner prints the wasm's SHA-256 on startup. Pass `--expect-sha256`
(or set `expected-sha256` in the config) to refuse anything else. Getting
the same hash on every machine also needs a pinned toolchain and trimmed
paths, which are not set up yet.

## Coordinator API

| Endpoint | Purpose |
|---|---|
| `POST /v1/join` | Register; returns a worker id and the run config |
| `GET /v1/task/{worker}` | Current round (410 when the run is over) |
| `POST /v1/slice/{worker}` | Lease the next slice no larger than `max_bytes` (204 if none) |
| `POST /v1/done/{worker}` | Slice ids fully trained on, after the update that used them was accepted |
| `GET /v1/weights/{round}` | Global weights as little-endian f32 |
| `GET /v1/data/{offset}/{len}` | Part of a `--corpus` |
| `POST /v1/update/{worker}/{round}?tokens=N` | int8-compressed `global - local` weights |
| `GET /v1/status` | Round, workers, slices leased and done, validation loss |

## Not done yet

- **Real model and tokenizer**: replace the byte-level bigram with a
  transformer and a BPE tokenizer. With `wasi:nn`, the parameters go to the
  graph and the gradient comes back on every step, which costs too much for
  large models. A graph that keeps the parameters and optimizer state on the
  device would avoid that.
- **Deduplication** across the dataset in the planner. Filters so far work on
  per-document columns only.
- **Spot checks**: replay a sample of a worker's steps (its slices and their
  order are known) and reject updates that don't match within tolerance.
- **GPU limits**: enforce `gpu-memory-mb` for `wasi:nn` graphs.
- **Reproducible worker builds**, so everyone gets the same hash.
- **Coordinator persistence and worker auth**: checkpoints, resume, rate limits.
- **Multiple coordinators** agreeing on each round's result, if no single
  operator should control the run.
