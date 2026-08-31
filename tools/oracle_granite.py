"""Run a converted Granite Q4NX model on the CPU, through transformers.

This is the oracle. It answers the question the NPU cannot yet answer -- whether
the conversion is *right* -- by putting the converted weights into somebody
else's implementation of the architecture and seeing what comes out. It needs no
NPU, no xclbin, and nothing from the FLM runtime.

Two checks, and they test different things:

1. **Fold equivalence.** The same Q4NX weights are loaded twice: once into a
   LlamaForCausalLM, which applies the stock `head_dim ** -0.5` attention scale
   -- exactly what FLM's llama engine does -- and once into a
   GraniteForCausalLM with the q_proj fold divided back out, which applies
   `attention_multiplier` itself. If the fold is correct these two must produce
   the same logits. This is the claim the whole port rests on, and it is checked
   against no external reference: same weights, two paths, must agree.

2. **Generation.** The Llama path generates text. A wrong RoPE permutation or a
   wrong tile layout does not produce a small numeric error, it produces
   garbage, so coherent output covers the parts of the conversion that a
   cosine diff against the source cannot reach.

Run (in an environment with torch + transformers):

    python tools/oracle_granite.py OUT_DIR --config path/to/granite/config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from einops import rearrange
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.unpack import dequantize_q4nx  # noqa: E402

# Keys FLM adds to config.json that transformers does not know about.
FLM_ONLY = {
    "addr_qk", "addr_kv", "addr_kk", "addr_l_begin_mha", "addr_l_end_mha",
    "flm_version", "vision_model_weight", "audio_model_weight", "is_vlm", "is_audio",
}


def load_q4nx_weights(model_dir: Path, cfg: dict) -> dict[str, torch.Tensor]:
    """Q4NX file -> a transformers state_dict, in HF layout.

    Undoes the two things the converter does to layout: the tile packing, and
    the RoPE permutation on q_proj/k_proj.
    """
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    vocab = cfg["vocab_size"]
    head_dim = cfg.get("head_dim") or hidden // heads

    shapes = {
        "self_attn.q_proj.weight": (heads * head_dim, hidden),
        "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
        "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
        "self_attn.o_proj.weight": (hidden, heads * head_dim),
        "mlp.gate_proj.weight": (inter, hidden),
        "mlp.up_proj.weight": (inter, hidden),
        "mlp.down_proj.weight": (hidden, inter),
    }

    packed = load_file(str(model_dir / "model.q4nx"))
    state: dict[str, torch.Tensor] = {}

    for name, tensor in packed.items():
        if tensor.dtype == torch.bfloat16:
            state[name] = tensor.float()  # embed_tokens, norms
            continue
        if name == "lm_head.weight":
            rows, cols = vocab, hidden
        else:
            suffix = name.split(".", 3)[-1] if name.startswith("model.layers.") else name
            if suffix not in shapes:
                print(f"[WARN] unknown packed tensor, skipping: {name}")
                continue
            rows, cols = shapes[suffix]

        w = dequantize_q4nx(tensor, rows, cols)
        if "q_proj" in name or "k_proj" in name:
            # Inverse of the converter's '(g p q) c -> (g q p) c'.
            w = rearrange(w, "(g q p) c -> (g p q) c", p=head_dim // 2, q=2).contiguous()
        state[name] = w

    return state


def build(cls, cfg_cls, cfg: dict, state: dict, dtype=torch.float32):
    config = cfg_cls(**{k: v for k, v in cfg.items() if k not in FLM_ONLY})
    model = cls(config).to(dtype)
    missing, unexpected = model.load_state_dict(
        {k: v.to(dtype) for k, v in state.items()}, strict=False
    )
    real_missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    if real_missing:
        print(f"[WARN] missing from state_dict: {real_missing[:8]}")
    if unexpected:
        print(f"[WARN] unexpected in state_dict: {unexpected[:8]}")
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--config", required=True, help="granite config.json (for dimensions)")
    ap.add_argument("--tokenizer", default="", help="dir holding tokenizer.json (default: model_dir)")
    ap.add_argument("--prompt", default="The capital city of France is")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from transformers.models.granite import GraniteConfig, GraniteForCausalLM

    model_dir = Path(args.model_dir)
    cfg = json.loads(Path(args.config).read_text())
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or hidden // heads
    attn_mult = cfg.get("attention_multiplier", head_dim ** -0.5)
    q_fold = attn_mult * (head_dim ** 0.5)

    print(f"[INFO] head_dim {head_dim}, attention_multiplier {attn_mult}, q_proj fold {q_fold}")

    state = load_q4nx_weights(model_dir, cfg)
    print(f"[INFO] loaded {len(state)} tensors from model.q4nx")

    # --- path A: the folded weights through a stock Llama (what FLM will do)
    llama_cfg = dict(cfg)
    for key in ("attention_multiplier", "embedding_multiplier",
                "residual_multiplier", "logits_scaling", "model_type",
                "architectures", "rope_parameters"):
        llama_cfg.pop(key, None)
    llama_cfg["head_dim"] = head_dim
    llama = build(LlamaForCausalLM, LlamaConfig, llama_cfg, state)

    # --- path B: fold divided back out, through the real Granite
    unfolded = dict(state)
    if q_fold != 1.0:
        for name in list(unfolded):
            if name.endswith("self_attn.q_proj.weight"):
                unfolded[name] = unfolded[name] / q_fold
    granite_cfg = dict(cfg)
    granite_cfg.pop("rope_parameters", None)
    granite_cfg.pop("architectures", None)
    granite = build(GraniteForCausalLM, GraniteConfig, granite_cfg, unfolded)

    tok_dir = args.tokenizer or str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        a = llama(ids).logits
        b = granite(ids).logits

    flat_a, flat_b = a.flatten().double(), b.flatten().double()
    cos = float(torch.dot(flat_a, flat_b) / (flat_a.norm() * flat_b.norm()))
    max_abs = float((a - b).abs().max())
    agree = int((a.argmax(-1) == b.argmax(-1)).all())
    print()
    print("[CHECK 1] fold equivalence -- llama(folded) vs granite(unfolded)")
    print(f"          logits cosine   : {cos:.8f}")
    print(f"          max abs diff    : {max_abs:.3e}")
    print(f"          same argmax     : {'yes' if agree else 'NO'}")
    fold_ok = cos > 0.9999 and agree

    print()
    print("[CHECK 2] generation through the llama path")
    with torch.no_grad():
        out = llama.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"          {text!r}")

    print()
    if not fold_ok:
        print("[FAIL] the fold is not equivalent -- the two paths disagree")
        return 1
    print("[PASS] fold equivalence holds; inspect the generation above for coherence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
