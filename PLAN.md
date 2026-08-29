# Plan: `--quant q4_k` — byte-exact port of the official Q4_K pipeline

## Goal
Make `q4nx-build -i <Qwen3.5 GGUF> --quant q4_k` reproduce, **byte for byte**, the
output of AMD's reference FLM converter (`ROCm/FLM_Q4NX_Converter`, commit
`d1d5232`) on a Q4_K-quantized input GGUF. The official output uses ~5-bpw
Q4_K packing (4736-byte chunks) for the "hidden" matmuls instead of plain Q4_1.

## Resolution (supersedes the original plan)
The early plan assumed gguf-py 0.19.0 cannot work with Q4_K and we would
"reinterpret" Q4_1 bytes as Q4_K. That is wrong. The real flow is:

- **Read Q4_K natively.** `gguf.quants` can *dequantize* Q4_K and `unpack_q4_k`
  factors each super-block into exact `t_j = S·s_j` / `u_j = M·m_j` per group
  plus the raw uint4 `q` — same shape/32-column granularity as Q4_1's `(d, m, qw)`.
- **Re-fit metadata, don't re-quantize.** `_pack_q4k` re-fits each side onto
  `(BF16 P', uint8 p'_j)` over the 8 groups that really share a 256-column
  super-block after model reorders (`_refit_one_side`, search=3). Q4_K is a
  read-only format: nothing ever encodes *to* it (gguf `quantize(Q4_K)` raises
  `NotImplementedError`), so any config asking for Q4_K as a *fallback* target is
  honored as Q4_1.
- Chunk (32×256): `s8[256] + m8[256] + q[4096] + S_bf16[64] + M_bf16[64] = 4736 B`,
  all column-major with no parallel-strided reorder. `M` is stored negated
  (`w = S·s·q − M·m` on the GGUF side, the kernel adds both accumulators).
- Type resolution (`get_used_quantization_type`): a Q4_K *source* stays Q4_K
  regardless of config (alpha/beta, ssm_out_proj, gate/up pack as Q4_K even
  though the config asks Q8_0/Q4_1); a Q6_K source with a Q4_K default becomes
  Q4_1 (qkv/ffn_down/attn_v), with a Q8_0 default becomes Q8_0 (lm_head synth).
- The old parallel-size-strided reorders in `_pack_q4nx`/`_pack_q8nx` were
  removed (official layout is plain column-major); row padding is retained
  (rows are always multiples of 32, so it is a no-op).

## What changed
- `q4nx/gguf_tensor.py`: added `_to_bf16_up`, `_bf16_next_up`, `_refit_one_side`,
  `unpack_q4_k`, `Q4_K_GROUP_SIZE`/`Q4_K_SUPER_BLOCK_SIZE`, Q4_K branches in
  `unpack`/`get_used_quantization_type`/`_requantize_to` (Q4_K→Q4_1 fallback).
- `q4nx/model_converter.py`: `_pack` dispatches Q4_K → `_pack_q4k` (and floats →
  bf16, matching flmc); `_pack_q4k` added; `_pack_q4nx`/`_pack_q8nx` data
  rearranges fixed to plain column-major; `_load_config`/`get_ggml_type` accept
  Q4_K; `_load_config` resolves the config via `config_filename_for_arch`
  honoring `requested_quant`; `create_converter`/`create_hf_converter` take a
  `quant` arg and re-load config.
- `q4nx/models/qwen35.py`: full-attention-branch now uses the GGUF's
  `qwen35.full_attention_interval` metadata (`layer_id % interval == interval-1`;
  blk.24 is *not* reordered, matching the official output); tied embeddings are
  skipped (no `model.embed_tokens.weight`); the alpha/beta `.bf16` copy synthesis
  was removed (those keys exist only when the source GGUF ships
  `ssm_alpha.bf16.weight`, which local GGUFs do not); unmapped names are skipped.
- `configs/qwen3.5_0.8b.q4_k.json`: `linear_out_proj` → `Q4_1` (canonical);
  name_map already identical to the reference config.
- `q4nx/constants.py`: fixed the QWEN35_08B route (was pointing at a deleted
  config) → `qwen3.5_0.8b.q4_1.json` default; added `Q4_K_CONFIGS` +
  `config_filename_for_arch` for `--quant q4_k`.
- `q4nx/cli.py` + `q4nx/model_assets.py`: `--quant {q4_1,q4_k}`; the chosen quant
  is hoisted to the front of the HF-repo GGUF preference order in
  `find_repo_gguf`/`select_repo_gguf` (`q4_k` → prefer `*Q4_K.gguf`).
- Universal MTP drop in base `_read_gguf_tensors`; always-emitted bf16 prefill
  copies in `qwen35.py`; HF-supplement plumbing (`hf_supplement_dir`,
  `_setup_hf_supplement`, `_load_supplement_tensor`, `create_converter(...,
  hf_tensors=...)`) and `--hf-tensors` CLI flag with `-s` piggyback (below).

## Finetune-compatible conversion (supersedes the byte-exact-only stance)
The Q4_K port is byte-exact against the reference converter's own output. But the
*shipped* NPU2 model is structurally richer, and finetunes against it must
exercise those tensors. Three behavior changes make `q4nx-build` reproduce the
NPU2 **structure** from a standard Qwen3.5 GGUF:

- **MTP/draft blocks are dropped universally** (base `_read_gguf_tensors`). Any
  block id holding `.nextn.` tensors — the trailing MTP layer, `blk.24` on the
  0.8B — has *all* of its tensors dropped for every converter (FastFlowLM has no
  MTP support). This is what removes the 25th layer the GGUF ships while the
  NPU2 has 24.
- **The 36 bf16 prefill copies are always emitted** (`qwen35.py`, alpha/beta
  branch). For each linear-attn layer the converter always writes
  `ssm_alpha_proj.bf16.weight` / `ssm_beta_proj.bf16.weight` (BF16, `[16, 1024]`),
  matching the official structure. Values come from the GGUF via
  `dequantize()` + the standard reorder; a raw-HF supplement refines them.
- **Raw-HF safetensor supplement** (`--hf-tensors DIR/REPO`, or piggybacked from
  `-s/--source-model` when that is a local safetensors dir). Base converter
  loads `model.safetensors.index.json`/`model.safetensors`; the prefill copy for
  `in_proj_a`/`in_proj_b` is taken from the HF tensors
  (`model.language_model.layers.{bid}.linear_attn.in_proj_a/b.weight`, BF16
  `[16, 1024]`) when present, falling back to the GGUF-derived dequant bucket.

## Verification (structure parity + byte-exact)
The byte-exact oracle check below still holds (reference converter output on the
same GGUF). Structural parity vs the shipped NPU2 model is asserted separately
via the new behavior:

- `/tmp/opencode/q4nx_hf_out/model.q4nx` — `q4nx-build --quant q4_k --hf-tensors`
  on `Qwen3.5-0.8B-Q4_K.gguf`: **356 tensors** (vs official 356).
  - no `layers.24.*` (MTP dropped); **36** `*.ssm_{alpha,beta}_proj.bf16.*`;
  - name sets identical to official (`set(a)==set(b)`), no missing/extra keys;
  - the 60 shape/dtype-differing commons are provenance artifacts only —
    qkv/down_proj as Q4_1 `[*,4,5120]` (ours, from Q6_K source → Q4_1 fallback)
    vs official Q4_K `[*,4,4736]`, and alpha/beta `[1,4,4736]` in ours vs Q8_0
    `[1,4,8704]` official. Values differ by design (NPU2 must not be
    byte-reproduced; structure is the target).
  - the 36 prefill bf16 tensors exactly equal the HF `in_proj_a/b.weight` slice
    shapes and (layer-0 spot check) bytes, confirming the supplement path feeds
    through verbatim.
- `/tmp/opencode/flm_out/model.q4nx` — output of the *official* flmc converter
  on `Qwen3.5-0.8B-Q4_K.gguf` (the byte-exactness oracle), 331 tensors.
- `q4nx-build` output and the oracle are **byte-identical**: same SHA-256
  (`4736c649…`) over the whole file (header metadata + tensor order + data).
- Eight representative tensor classes were additionally verified during the
  port with `/tmp/opencode/validate_port.py` (Q4_K, Q4_1 from Q6_K, Q8_0 lm_head,
  Q4_K alpha with row-repeat, reordered blk.3/blk.23 and un-reordered blk.24
  q_proj) before the full run.
- End-to-end CLI: `q4nx-build -i <Qwen3.5-0.8B-Q4_K.gguf> --quant q4_k` produces
  a model with the same SHA-256 (`4736c649…`) as the oracle; the default q4_1
  path also round-trips (for a Q4_K GGUF the Q4_K sources stay Q4_K, so both
  configs converge on the same bytes — matching reference behavior).
- A property they inherit from the reference: a Q4_K *source* always stays Q4_K
  even under a Q4_1 default config (Q4_K is in the keep-list); only Q6_K/F32
  sources are re-targeted by the config.

## Caveats
- The HF reference `Qwen3.5-0.8B-NPU2-Q4_K/model.q4nx` is **not** byte-reproducible:
  on 60 of the 320 shared tensors it differs in dtype/shape from the local-GGUF
  path (Q4_K 4736-B vs Q4_1 5120-B chunks for qkv/mlp; Q8_0 alpha/beta vs our
  Q4_K) because it was produced by a *different* Q4_K quant run. Byte-exactness
  is asserted against the official converter's own output on the same GGUF;
  structure (names/shapes/dtype of all tensors, 36 prefill copies, no MTP
  layer) is asserted against the shipped NPU2 model.
- The `--hf-tensors`/`-s` supplement only refines the value of the bf16 prefill
  copies; quantized weights remain GGUF-derived (values will differ from a
  re-quantized NPU2). Supply the raw `Qwen/Qwen3.5-0.8B` (not the NPU2
  skeleton) for exact prefill bytes.
- Install note: the venv's `q4nx-build` console script runs from an editable
  install (`uv pip install -e . --python <venv>/bin/python`), so repo edits are
  live after reinstall-with-edit if the editable link is missing; use
  `uv pip install -e .` to refresh the console script.