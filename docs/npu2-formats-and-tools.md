# NPU2 Formats & Tooling Guide

**Audience:** developers working on FastFlowLM (AMD ROCm), XRT/mlir-aie, or
q4nx-build converters. No prior knowledge of this repo assumed.

This document describes the on-disk and in-memory formats that carry a
quantized LLM from conversion to NPU execution on AMD Ryzen AI "NPU2"
(Phoenix/Strix/Krackan AIE) hardware, and the tools in this repo that parse
them. Everything here was derived from public sources (XRT headers,
mlir-aie, FastFlowLM's `npu_cmd*.hpp`) plus byte-level analysis of shipping
artifacts; confidence labels are given per section.

Related: [BREAKING-CHANGES.md](BREAKING-CHANGES.md) — dated log of upstream
updates that moved one of these formats, with triage notes.

---

## 1. The big picture

Three artifacts cooperate at inference time:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │ model.q4nx            safetensors container holding quantized    │
 │                       weights in NPU-tiled block layouts          │
 ├──────────────────────────────────────────────────────────────────┤
 │ *.xclbin              AXLF container: kernel metadata + CDO/PDI   │
 │                       blob that statically configures every AIE   │
 │                       tile (DMA BDs, stream switch, DM constants) │
 ├──────────────────────────────────────────────────────────────────┤
 │ instruction buffer    runtime transaction stream (built per-step  │
 │ (transaction buffer)  by the app, e.g. FastFlowLM npu_sequence),  │
 │                       shipped as an ELF to XRT; programs shim DMA │
 │                       transfers and kicks the DPU                 │
 └──────────────────────────────────────────────────────────────────┘
```

A conversion tool (this repo's `convert.py`) only produces the first
artifact. It stays correct as long as the *weight layouts* the compiled
kernels expect don't move. The other two artifacts are produced by AMD's
toolchain and change whenever kernels are rebuilt — which is why triage
(section 7) matters.

## 2. AXLF / xclbin2 container  *(high confidence)*

Defined in XRT's [`xclbin.h`](https://github.com/Xilinx/XRT/blob/master/src/runtime_src/core/include/xclbin.h).
Parser: `tools/axlf.py`.

File header (little-endian):

| offset | size | field |
|-------:|-----:|-------|
| `0x000` | 8   | magic `"xclbin2\0"` |
| `0x008` | 4   | signature length (`-1` = none) |
| `0x00C` | 28  | reserved |
| `0x028` | 256 | key block |
| `0x128` | 8   | unique id |
| `0x130` | …   | `axlf_header` (timestamps, version, platform VBNV, featureRom UUID) |
| `0x1A0` | 16  | **UUID** |
| `0x1C0` | 4   | section count |
| `0x1C8` | …   | section table |

Section table entry (40 bytes each):

| offset | size | field |
|-------:|-----:|-------|
| `+0x00` | 4  | section kind |
| `+0x04` | 16 | name string |
| `+0x18` | 8  | offset from file start |
| `+0x20` | 8  | size |

Kinds seen in NPU2 xclbins: `MEM_TOPOLOGY (0x06)`, `EMBEDDED_METADATA
(0x02)` — the connectivity XML carrying the kernel name and argument list —,
`IP_LAYOUT (0x07)`, `AIE_PARTITION (0x19)`, and the big one,
`AIE_PARTITION_PDI (0x20)`, which holds the CDO payload described next.

## 3. CDO v2 — static AIE configuration  *(high confidence)*

Parser: `tools/cdo.py`. The PDI section starts with ~0x21C bytes of wrapper;
the CDO proper begins there:

```
[0x21C] 0xFDFB4175   framing marker
[0x220] 0x00000004   framing count
[0x224] 0x004F4443   magic "CDO\0"
[0x228] 0x00000200   version 2.0
[0x22C] total length (words)
[0x230] checksum
[0x234] first command word
```

Every command word: `opcode bits[7:0] · module_id bits[15:8] · payload_len
bits[23:16]` (payload_len `0xFF` → long form, real length in next word).

| opcode | name | payload |
|-------:|------|---------|
| `0x02` | MASK_WRITE  | `[addr, mask, value]` |
| `0x03` | WRITE       | `[addr, value]` |
| `0x05` | DMAWRITE    | `[addr_hi, addr_lo, data…]` |
| `0x07` | MASKWRITE64 | `[hi, lo, mask, value]` |
| `0x08` | WRITE64     | `[hi, lo, value]` |
| `0x11` | NOP         | — |
| `0x01FF` | END       | full-word marker |

**AIE2 tile addressing:** `addr = col<<25 | row<<20 | local_offset`
(row 0 = shim, 1 = memory tile, 2–5 = compute cores).

Core-tile local map used throughout the tooling:

| range | region |
|-------|--------|
| `0x10000–0x1CFFF` | program memory (rarely loaded statically) |
| `0x1D000–0x1FFFF` | core DMA buffer descriptors |
| `0x20000–0x2FFFF` | data memory (static constants *and* the DPU control program) |
| mem-tile `0x00000–0x7FFFF` | 512 KB SRAM |
| mem-tile `0xA0000–0xAFFFF` | memory-tile DMA BDs |

Empirical note: in shipping NPU2 kernels the core-tile DM at `0x20000`
contains the DPU control program plus config words, not model weights —
weights live in `model.q4nx` and arrive via DDR. When AMD rebuilds a
kernel, this block changes shape (see BREAKING-CHANGES).

## 4. Transaction buffer — runtime control stream  *(high confidence for FLM native)*

Disassembler: `tools/txn.py`. Two encodings exist in the wild.

### 4.1 FastFlowLM native `npu_seq`

Source of truth: `FastFlowLM/src/include/npu_utils/instr_utils/npu_cmd*.hpp`
and the `seq2cmds` dispatcher in `npu_instr_utils.hpp`.

4-word device header, then commands back-to-back:

```
[0] major[7:0] | minor[15:8] | dev_gen[23:16] | rows[31:24]
[1] cols[7:0]  | mem_tile_rows[15:8]
[2] instruction count
[3] stream size in bytes
```

| opcode | name | words | layout |
|-------:|------|------:|--------|
| `0x00` | WRITE | 6 | `[op][0][addr][0][value][size*4]` |
| `0x01` | BLOCKWRITE | 12 | shim DMA BD config (below) |
| `0x03` | MASKWRITE | 7 | `[op][0][addr][0][value][mask][size*4]`; issue-token form targets a queue register with `mask=0x1F00`, `value=pkt_id<<8` |
| `0x05` | NOOP | 1 | |
| `0x06` | PREEMPT | 1 | level in bits `[9:8]` |
| `0x80` | TCT (wait-sync) | 4 | `[op][16][row<<8\|col<<16\|dir_bit0][ch<<24\|0x10100]` |
| `0x81` | DDR_PATCH | 12 | `[op][48<<2][0×4][bd_reg][0][arg_idx][0][arg_offset][0]` |

**Shim DMA BD config** (`BLOCKWRITE`, from `npu_dma_block_cmd::to_npu`):

```
w[2]  = row<<20 | col<<25 | bd_id<<5 | 0x1D000      # target BD register
w[3]  = 48                                          # op_size*4
w[4]  = buffer_length (words)
w[5]  = buffer_offset
w[6]  = pkt_en<<30 | ooo_id<<24 | pkt_id<<19 | pkt_type<<16
w[7]  = d0_size<<20 | (d0_stride-1)                 # 0 ⇒ linear transfer
w[8]  = burst(0xC)<<30 | d1_size<<20 | (d1_stride-1)
w[9]  = axcache<<24 | (d2_stride-1)
w[10] = (iter_size-1)<<20 | (iter_stride-1)
w[11] = next_bd<<27 | use_next<<26 | valid<<25 | lock fields
```

Strides/iters are stored **minus one**; FLM clamps them to ≥1 before
encoding, so an "unset" dimension is `0` on the wire, not `0xFFFFFFFF`.

**Queue push** (a `WRITE` whose register satisfies `(reg & 0x1FE00)==0x1D200`)
kicks a BD:

| reg | meaning |
|-----|---------|
| `0x1D200/04` | S2MM ch0 |
| `0x1D208/0C` | S2MM ch1 |
| `0x1D210/14` | MM2S ch0 |
| `0x1D218/1C` | MM2S ch1 |

`value = bd_id | repeat<<16 | issue_token<<31`. MM2S reads DDR→array
(host-to-device); S2MM streams array→DDR (device-to-host).

**DDR_PATCH** ties a BD to a kernel argument: `bd_reg = col<<25 | row<<20 |
bd_id<<5 | 0x1D004`; XRT patches the BD's DDR address from BO `arg_idx` at
byte `arg_offset`.

### 4.2 aiebu DPU ctrltext

After aiebu assembles the ELF delivered to `xrt::elf`, BD programming can
appear as `DMA_BD_FENCE` (opcode `0x30`) followed by six `WRITE`s.
`txn.py find_dma_bd_groups()` decodes this variant; `RELA` relocations
patch the DDR address placeholder in the first write.

## 5. Q4NX weight container  *(high confidence; verified numerically)*

A `model.q4nx` is a standard **safetensors** file. Quantized tensors are
stored as int8 arrays of packed blocks shaped `(R, C, block_bytes)` where
each block covers a `32 × 256` tile of the logical weight matrix:

| type | block bytes | layout inside a block |
|------|------------:|-----------------------|
| Q4_1 | 5120 | scales+dmins (bf16, per column-group of 32) then nibbles |
| Q8_0 | 8704 | scales (bf16, per column-group of 32) then int8 quants |

The mapping from GGUF/HF tensor names to Q4NX names involves head-tiling
transforms (`_untile_*` in `q4nx/models/qwen35moe.py`). The independent
decoders in `tools/verify_q4nx_quant.py` re-derive the pack layout from
first principles and cosine-compare against re-quantized references — if
that passes at ≥0.99, the block order/scale placement is right regardless
of what changed elsewhere.

## 6. Tool reference

All Python tools run under the repo venv; parsers take raw bytes/paths and
are import-safe as libraries.

| tool | purpose |
|------|---------|
| `axlf.py` | AXLF container parser (sections, connectivity XML, mem topology) |
| `cdo.py` | CDO v2 command parser; extracts tile programs / static DM writes |
| `txn.py` | transaction-buffer disassembler + DMA topology; CLI included |
| `ff_instr_dump.c` | LD_PRELOAD shim capturing `xrt::elf` payloads (`make` in `tools/`) |
| `xclbin_inspect.py` | interactive xclbin inspection: sections, args, CDO summary, PM dump, two-file diff |
| `xclbin_dm_analyze.py` | deep dive into one xclbin's core-tile DM |
| `xclbin_dm_diff.py` | segment-level DM diff between two xclbins or two git refs |
| `xclbin_catalog.py` | catalog every xclbin under a tree → JSON/markdown |
| `flm_triage.py` | verdict engine for upstream updates (next section) |
| `detect_family.py` | GGUF architecture detection heuristics chart |
| `verify_moe_q4nx.py` | float-level verification vs GGUF/HF reference |
| `verify_q4nx_quant.py` | numeric verification of packed Q4_1/Q8_0 blocks |

Capture workflow example:

```bash
cd tools && make
mkdir -p /tmp/ff_instr
FF_DUMP_INSTR_PATH=/tmp/ff_instr LD_PRELOAD=$PWD/ff_instr_dump.so flm run qwen3:4b
# pick the instr section out of a captured ELF, then:
venv/bin/python tools/txn.py buf.bin            # disassemble
venv/bin/python tools/txn.py buf.bin --json     # machine-readable
```

## 7. Triaging a new FastFlowLM release

```bash
git -C ../FastFlowLM fetch --tags
venv/bin/python tools/flm_triage.py vOLD vNEW            # full analysis
venv/bin/python tools/flm_triage.py vOLD vNEW --skip-xclbins   # 60 ms source-only pass
venv/bin/python tools/flm_triage.py vOLD vNEW --model GPT-OSS-20B-NPU2
```

Verdicts (also the exit code):

| verdict | meaning | action |
|---------|---------|--------|
| `OK` | cosmetic | nothing |
| `REVIEW` | model-runtime code changed | read diffs; rerun verify tools on a converted model |
| `KERNELS` | xclbins rebuilt | compare structure via the tool's output; conversions usually still load if interfaces match |
| `FORMAT` | `npu_utils/**` or `tensor_utils/**` changed | re-derive `txn.py`/`cdo.py`/pack layouts before converting anything |

Drill-down after a KERNELS verdict:

```bash
venv/bin/python tools/xclbin_dm_diff.py --repo ../FastFlowLM vOLD vNEW \
    --model Qwen3.6-35B-A3B-NPU2 --kernel mm.xclbin --tile 0,2
```

Then record the outcome in [BREAKING-CHANGES.md](BREAKING-CHANGES.md).

## 8. Provenance

| claim | source |
|-------|--------|
| AXLF offsets/kinds | XRT `xclbin.h`; validated against 209 shipping xclbins |
| CDO opcodes/framing | XRT `cdo_cmd.h` via aie-pdi-transform; validated on all FLM kernels |
| txn opcodes/layouts | FLM `npu_cmd*.hpp` + `npu_instr_utils.hpp` (public repo); round-trip tested |
| aiebu fence variant | empirical (captured ELFs); treat as best-effort |
| Q4NX block layouts | independently decoded + cosine-verified against gguf-py re-quantization |
