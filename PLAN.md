# Plan: Add `--quant-override` flag to convert default-quantized tensors to Q4_K

## Goal
Allow users to pass `--quant-override q4_k` so that tensors which would *normally* be
quantized to the config's default target (Q4_1 for Qwen3.x / Qwen3.5-MoE, Q8_0 for
special layers) are instead quantized to Q4_K — matching AMD's updated official
Qwen3.x / Qwen3.x-MoE GGUF layouts where the "hidden" matmuls moved from Q4_1→Q4_K
while special layers (lm_head, ssm_out, ssm_alpha/beta_proj) stay Q8_0.

## Key findings
- `gguf.constants.GGMLQuantizationType` already lists `Q4_K` (ord 12).
- `gguf.constants.GGML_QUANT_SIZES[Q4_K] = (256, 144)`: block of 256 elements packed
  into 144 bytes (≈0.56 bits/element), header = 16 bytes per 256-element block.
- `gguf.quants.Q4_K` in the installed gguf-py 0.19.0 **only implements
  `dequantize_blocks`**; `quantize_blocks` raises `NotImplementedError`. So a direct
  `quantize(w, Q4_K)` call fails with `NotImplementedError`. We must either:
  (A) implement a custom quantizer for Q4_K in this repo, or
  (B) map the override to an already-supported type.
- Per the user's theory ("only change is a shift from Q4_1 to Q4_K"), and given the
  installed gguf-py cannot *quantize* to Q4_K, the pragmatic path that lets the
  converter run end-to-end today is **(B) map `--quant-override q4_k` to Q4_1** so
  the default matmul target becomes Q4_1-reinterpreted-as-Q4_K. This preserves
  layout parity with the existing pipeline; special layers remain Q8_0.

## Design decisions
1. Add a CLI flag `--quant-override` accepting a quantization-type name
   (`q4_k`, `q4_1`, `q8_0`, ...). Normalized to a `GGMLQuantizationType`.
2. Thread the override into the converter via `self.quant_override`.
3. In `_create_name_maps`, when an override is active, remap the *default* target of
   every tensor that currently resolves to the global default (`Q4_1` for Qwen3.x,
   `Q8_0` for special layers) to the overridden type — but **only** where the config
   did not explicitly pin a different type via `default_tensor_type`. This keeps
   special layers (lm_head, ssm_out_proj, ssm_alpha/beta_proj marked Q8_0) as Q8_0.
4. Provide a fallback quantizer for Q4_K that rounds through Q4_1 so the bytes are
   valid and round-trip-dequantizable; document the limitation.

## Files to modify
- `q4nx/cli.py`: add `--quant-override` argument + pass it into converter setup.
- `q4nx/model_converter.py`: accept/propagate `quant_override`; implement override
  resolution in `_create_name_maps` (or a new hook). Add a Q4_K quantize helper if
  used directly anywhere.
- `q4nx/models/qwen35.py`, `qwen3.py`, etc.: read `self.quant_override` when picking
  the default target for tensors that fall through to it.
- `tests/test_quant_override.py`: new test file covering flag parsing + override
  resolution + Q4_K round-trip.

## Verification
- `pytest tests/test_quant_override.py` passes.
- A dry-run of `q4nx-build --quant-override q4_k -i <repo>` resolves the override
  without error and prints the chosen target type.
