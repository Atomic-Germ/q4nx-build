# FastFlowLM Breaking-Change Log

Dated, append-only record of upstream updates that moved something
q4nx-build depends on. One section per release pair, newest last.

Triage verdicts come from `tools/flm_triage.py` (see
[npu2-formats-and-tools.md](npu2-formats-and-tools.md) §7):

`OK` < `REVIEW` < `KERNELS` < `FORMAT`

---

## v1.0.1 → v1.0.2  (triaged 2026-08)

**Verdict: KERNELS** — no format changes; kernels rebuilt, Qwen3.5-family
mm-path heavily rewritten.

### What changed

**FORMAT surfaces — clean.**
Zero diffs in `src/include/npu_utils/**` (transaction command encoding) and
`src/include/tensor_utils/**` (`Q4NX`/`SafeTensors` container). Kernel
interfaces (arg names/types/order) unchanged on every xclbin.

**xclbins:** 174/209 shared files byte-identical; 35 changed; 9 added
(entire new `Qwen3.5-OMNI-NPU2` set).

Two distinct patterns among the changed kernels:

1. **Dense `layer.xclbin`s** (DeepSeek/Llama/Qwen3/Phi4/Nanbeige families):
   uniform +2,336 B file growth. Core-tile DM grew +2,352 B across 17 tiles
   per kernel — sixteen tiles exactly **+128 B**, the column-local tile
   `(2,2)` by **+304 B**. The first DM segment (the DPU control program at
   local offset `0x20000`) absorbed the growth; all later segments shifted
   up by `0x80` with content otherwise intact. Reads as a mechanical
   recompile of the control program, not a layout change.

   ```
   tile(0,2): 5 segs/9208 B -> 5 segs/9336 B (+128)
     ≡ 1988 B  (shift 0x21870 -> 0x218f0)
     ~ 6256 -> 6384 B @ 0x20000  (program block grew)
   ```

2. **Qwen3.5/3.6 MoE mm path** (`mm.xclbin`, `dequant_mm.xclbin`,
   `lm_head.xclbin`): full resegmentation, static DM roughly halved
   (e.g. Qwen3.6 `mm.xclbin` 266 KB → 127 KB static DM; file 330 KB →
   187 KB). Matches upstream commit `376c98a` "huge optimization to
   qwen3.5 fam" / `fa38c05` "update optimized qwen3.5/3.6 dll/so".
   Same tiles touched, same interfaces — internals rescheduled.

**REVIEW surface:** `modeling_*.cpp` changes are image-handling robustness
(base64 validation before placeholder insertion), not weight loading.
`lm_config.hpp` refactor extracts `cfg_sub()` — behavior-neutral.

### Impact on q4nx-build

Converter output remains structurally valid: container format, block
layouts, and tensor naming untouched. The risk area is *behavioral* — the
rewritten MoE mm/dequant kernels may consume their DM config words or
instruction streams differently. If a converted model misbehaves on ≥1.0.2,
diff its kernel's DM segments against the tag it was built for:

```bash
venv/bin/python tools/xclbin_dm_diff.py --repo ../FastFlowLM \
    v1.0.1 v1.0.2 --model Qwen3.6-35B-A3B-NPU2 --kernel mm.xclbin
```

### Reproduce

```bash
venv/bin/python tools/flm_triage.py v1.0.1 v1.0.2
```
