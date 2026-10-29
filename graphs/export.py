"""Exports a dti model as a TorchScript graph that dti-runner can run through wasi:nn.

    python graphs/export.py byte-bigram byte-bigram.pt

Use torch 2.4 to match the libtorch the runner's PyTorch backend links. The
graph takes `params: f32[n]` and `tokens: u8[t]`, and returns a single tensor
`[loss, grad...]` because the backend gives the worker one output. Loss and
gradient must match the Rust model in crates/model of the same name.
"""

import hashlib
import io
import sys

import torch


class ByteBigram(torch.nn.Module):
    """`dti_model::ByteBigram`: params[a * 256 + b] is the logit for byte b after a."""

    def forward(self, params: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        vocab = 256
        torch._assert(params.dim() == 1 and params.size(0) == vocab * vocab, "params must be f32[65536]")
        torch._assert(tokens.dim() == 1 and tokens.size(0) >= 2, "need at least 2 tokens")
        # Grad mode is on: the runner does not call graphs under no_grad.
        params = params.detach().requires_grad_()
        tokens = tokens.long()
        logits = params.view(vocab, vocab)[tokens[:-1]]
        loss = torch.nn.functional.cross_entropy(logits, tokens[1:])
        grad = torch.autograd.grad([loss], [params])[0]
        assert grad is not None
        return torch.cat([loss.detach().reshape(1), grad])


MODELS = {"byte-bigram": ByteBigram}


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in MODELS:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(MODELS)}> <out.pt>")
    name, out = sys.argv[1:]
    # Saving to a buffer keeps the output file name out of the archive (and the hash).
    buf = io.BytesIO()
    torch.jit.save(torch.jit.script(MODELS[name]()), buf)
    with open(out, "wb") as f:
        f.write(buf.getvalue())
    sha256 = hashlib.sha256(buf.getvalue()).hexdigest()
    print(f"wrote {out} (torch {torch.__version__}); add to the runner config:\n")
    print(f'[graphs.{name}]\npath = "{out}"\nsha256 = "{sha256}"\ntarget = "cpu"  # or "gpu" (CUDA)')


if __name__ == "__main__":
    main()
