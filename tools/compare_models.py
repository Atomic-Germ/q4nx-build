"""Compare two converted Q4NX models by running both and diffing the logits.

Written for one question: **is a padded model still the same model?**
`--pad-hidden` widens the residual stream onto a width the engine has a design
for. Every padded lane should stay identically zero, and RMSNorm's width term is
corrected so the wider mean cancels. If that reasoning is right, the padded
model and the original must agree on logits to within bf16 noise -- and if it is
wrong, they will not, which is the whole point of checking before spending time
on hardware.

Both models are loaded as `LlamaForCausalLM`, which is what FLM's llama engine
is: the stock `head_dim ** -0.5` attention scale, no Granite multipliers (the
converter folded those into the weights).

    python tools/compare_models.py DIR_A DIR_B --tokenizer DIR_A

They are built and freed one at a time -- a 3B model is ~6.4 GB at bf16.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_granite import FLM_ONLY, load_q4nx_weights  # noqa: E402


def logits_for(model_dir: Path, ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or hidden // heads

    state = load_q4nx_weights(model_dir, cfg, dtype)

    llama_cfg = {k: v for k, v in cfg.items() if k not in FLM_ONLY}
    for key in ("attention_multiplier", "embedding_multiplier", "residual_multiplier",
                "logits_scaling", "model_type", "architectures", "rope_parameters",
                "q4nx_folded_multipliers", "q4nx_padded_from_hidden",
                "q4nx_pad_norm_fix"):
        llama_cfg.pop(key, None)
    llama_cfg["head_dim"] = head_dim

    # transformers validates `hidden_size % num_attention_heads == 0` at
    # construction. A padded model breaks that on purpose -- granite is 40
    # heads of 64 in a 3072-wide residual stream -- and the check is about the
    # usual case where head_dim is derived, which is exactly what padding
    # stops being true. The validator only runs in __init__, so build at the
    # unpadded width and widen afterwards, before the layers are created.
    config = LlamaConfig(**{**llama_cfg, "hidden_size": heads * head_dim})
    config.hidden_size = hidden

    model = LlamaForCausalLM(config).to(dtype)
    missing, unexpected = model.load_state_dict(
        {k: v.to(dtype) for k, v in state.items()}, strict=False)
    real_missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    if real_missing:
        print(f"[WARN] missing: {real_missing[:6]}")
    if unexpected:
        print(f"[WARN] unexpected: {unexpected[:6]}")
    model.eval()
    with torch.no_grad():
        out = model(ids).logits.float()
    del model, state
    gc.collect()
    print(f"[INFO] {model_dir.name}: hidden {hidden}, head_dim {head_dim}, "
          f"eps {llama_cfg.get('rms_norm_eps')}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--prompt", default="The capital city of France is")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    from transformers import AutoTokenizer

    a_dir, b_dir = Path(args.dir_a), Path(args.dir_b)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or str(a_dir))
    ids = tok(args.prompt, return_tensors="pt").input_ids
    dtype = getattr(torch, args.dtype)

    a = logits_for(a_dir, ids, dtype)
    b = logits_for(b_dir, ids, dtype)

    fa, fb = a.flatten().double(), b.flatten().double()
    cos = float(torch.dot(fa, fb) / (fa.norm() * fb.norm()))
    max_abs = float((a - b).abs().max())
    agree = bool((a.argmax(-1) == b.argmax(-1)).all())
    top_a = tok.decode(a[0, -1].argmax())
    top_b = tok.decode(b[0, -1].argmax())

    print()
    print(f"  logits cosine  : {cos:.8f}")
    print(f"  max abs diff   : {max_abs:.3e}")
    print(f"  same argmax    : {'yes' if agree else 'NO'}")
    print(f"  next token     : {top_a!r} vs {top_b!r}")
    print()
    if cos > 0.9999 and agree:
        print("[PASS] the two models agree")
        return 0
    print("[FAIL] the two models disagree")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
