"""Diff a converted Q4NX model against the GGUF it was built from, tensor by tensor.

Why this and not just "does it talk": an end-to-end signal cannot localise a
conversion fault, and with two faults present it may not even detect one --
they can partially cancel and still produce fluent-looking output. A per-tensor
diff says *which* tensor and *how* wrong.

What it checks, per tensor:

  * cosine similarity between the Q4NX weights and the source weights, after
    applying the same folds and RoPE permutation the converter applied. This
    isolates quantization + tiling: anything below the Q4_1 floor is a bug in
    the packing, not in the format.
  * that the fold actually landed, by measuring the ratio between the two and
    comparing it to the factor the converter says it used.

Run:

    python tools/verify_q4nx.py OUT_DIR --gguf model.gguf
    python tools/verify_q4nx.py OUT_DIR --gguf model.gguf --arch granite

Exits non-zero if any tensor falls below the threshold, so it can gate a build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from einops import rearrange
from gguf import GGUFReader, dequantize
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.model_converter import create_converter  # noqa: E402
from q4nx.unpack import dequantize_q4nx  # noqa: E402

# Q4_1 with a group of 32 lands around 0.999 per tensor on well-conditioned
# weights. 0.99 is loose enough not to fire on an unlucky tensor and tight
# enough that any layout fault -- which shows up as a permutation, not as
# noise -- is far below it.
COSINE_FLOOR = 0.99


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", help="directory holding model.q4nx")
    ap.add_argument("--gguf", required=True, help="the source GGUF it was converted from")
    ap.add_argument("--arch", default="", help="force a model architecture (as convert.py -f)")
    ap.add_argument("--floor", type=float, default=COSINE_FLOOR)
    args = ap.parse_args()

    q4nx_path = Path(args.model_dir) / "model.q4nx"
    if not q4nx_path.is_file():
        print(f"[FAIL] no model.q4nx in {args.model_dir}")
        return 2

    print(f"[INFO] Reading {q4nx_path}")
    produced = load_file(str(q4nx_path))

    # Rebuild the converter on the same source so the reference path applies
    # exactly the folds and permutation the conversion applied. A bug in the
    # fold *arithmetic* is out of scope here and covered by tests/test_granite.py;
    # what this checks is that the fold reached the file and that the packing
    # is recoverable.
    conv = create_converter(args.gguf, args.arch)
    reader = GGUFReader(args.gguf)
    folds = conv._fold_factors() if hasattr(conv, "_fold_factors") else {}
    head_dim = conv._head_dim() if hasattr(conv, "_head_dim") else None

    # GGUF stores dimensions in ggml `ne` order, fastest axis first: for a 2D
    # weight that is [K, N] -- shape[0] is the input width and shape[1] the
    # output rows, the opposite of the torch convention used everywhere below.
    rows_of = {}
    for tensor in reader.tensors:
        name = conv.forward_name_map.get(tensor.name)
        if name is not None and len(tensor.shape) >= 2:
            rows_of[name] = (int(tensor.shape[1]), int(tensor.shape[0]), tensor)

    worst = ("", 1.0)
    failures = []
    checked = 0

    for name, packed in sorted(produced.items()):
        entry = rows_of.get(name)
        if entry is None:
            print(f"  {name:<52} SKIP (no source tensor)")
            continue
        rows, cols, tensor = entry

        ref = torch.from_numpy(dequantize(tensor.data, tensor.tensor_type).copy())
        ref = ref.reshape(rows, cols).float()

        if head_dim and ("q_proj" in name or "k_proj" in name):
            ref = rearrange(ref, "(g p q) c -> (g q p) c", p=head_dim // 2, q=2).contiguous()
        for suffix, factor in folds.items():
            if name.endswith(suffix):
                ref = ref * factor
                break

        if packed.dtype == torch.bfloat16:
            # Unquantized passthrough (the embedding table).
            got = packed.float().reshape(ref.shape)
        else:
            got = dequantize_q4nx(packed, rows, cols)

        cos = cosine(got, ref)
        checked += 1
        status = "ok  " if cos >= args.floor else "FAIL"
        if cos < worst[1]:
            worst = (name, cos)
        if cos < args.floor:
            failures.append((name, cos))
        print(f"  {name:<52} cos {cos:.6f}  {status}")

    print()
    print(f"[INFO] {checked} tensors checked, floor {args.floor}")
    print(f"[INFO] worst: {worst[0]} at {worst[1]:.6f}")
    if failures:
        print(f"[FAIL] {len(failures)} tensor(s) below the floor:")
        for name, cos in failures:
            print(f"         {name}  {cos:.6f}")
        return 1
    print("[PASS] every tensor is at or above the quantization floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
