"""IBM Granite (dense) -> Q4NX.

Granite's dense checkpoints are Llama with four scalar multipliers applied at
fixed points in the forward pass. Everything else -- tensor names, GQA, RoPE,
SwiGLU, RMSNorm -- is Llama's, and `gguf`'s own tensor set for `granite` is
exactly Llama's dense subset. So this subclasses :class:`Llama` and adds only
the multipliers.

**All four fold exactly into the weights**, which is what makes Granite runnable
on FLM's stock llama engine without a runtime change (nothing in the FLM runtime
reads `attention_multiplier`). Reading the HF `GraniteAttention` /
`GraniteModel` forward pass, and writing `hd` for head_dim:

    attention_multiplier   replaces llama's implicit hd**-0.5 attention scale
                           -> q_proj *= attention_multiplier * sqrt(hd)
    embedding_multiplier   inputs_embeds = embed(ids) * embedding_multiplier
                           -> embed_tokens *= embedding_multiplier
    residual_multiplier    h = residual + h * residual_multiplier, after BOTH
                           the attention block and the MLP block
                           -> o_proj *= residual_multiplier
                              down_proj *= residual_multiplier
    logits_scaling         logits = lm_head(h) / logits_scaling
                           -> lm_head /= logits_scaling

Folding into a *quantized* tensor is lossless, and cheaper than folding into the
float weights before quantization. A Q4_1 block stores `w = code * d + m` with
`d = (max-min)/15` and `m = min`; scaling every weight in the block by a
constant `c > 0` scales `max` and `min` by `c`, hence `d` and `m` by `c`, and
leaves every 4-bit code untouched. So scaling `(d, m)` after quantization gives
bit-identical codes to quantizing `c*W` directly. For granite-4.2-3b the one
non-unit multiplier is `attention_multiplier = 0.015625` against `hd = 64`,
giving `c = 0.125` -- a power of two, so even the `d`/`m` scaling is exact.

RoPE commutes with the q_proj fold: RoPE is a rotation, and a rotation of a
scaled vector is the scaled rotation of it.
"""

from __future__ import annotations

import math

import torch
from gguf import GGUFReader

from ..constants import ModelArch
from .llama import Llama


class Granite(Llama, model_arch=ModelArch.GRANITE):
    # GGUF metadata keys, tried in order. A GGUF converted under `-f granite`
    # from a file whose own arch string is `llama` still carries `llama.*`
    # keys, so both prefixes are searched.
    _ARCH_PREFIXES = ("granite", "llama")

    # Set by --pad-hidden / --pad-intermediate / --no-pad-norm-fix. See _pad_plan.
    pad_hidden: int | None = None
    pad_intermediate: int | None = None
    pad_norm_fix: bool = True

    # Which axis of each tensor carries the hidden dimension. Names are q4nx
    # names; matching is on suffix so it holds across layers.
    _PAD_COLS = (  # hidden is the INPUT width
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "mlp.gate_proj.weight",
        "mlp.up_proj.weight", "lm_head.weight",
    )
    _PAD_ROWS = (  # hidden is the OUTPUT width
        "self_attn.o_proj.weight", "mlp.down_proj.weight",
    )
    _NORMS = (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "model.norm.weight",
    )
    # Which axis of each tensor carries the INTERMEDIATE (MLP) dimension.
    # Padding this axis is exact with nothing to correct: the padded lanes of
    # gate/up are zero, `silu(0) * 0 = 0`, and down_proj's matching columns are
    # zero too. No norm spans the intermediate axis, so unlike --pad-hidden
    # there is not even a bf16 rounding.
    _PAD_INTER_ROWS = ("mlp.gate_proj.weight", "mlp.up_proj.weight")
    _PAD_INTER_COLS = ("mlp.down_proj.weight",)

    def _meta(self, suffix: str):
        """Read `<arch>.<suffix>` from the GGUF metadata, or None."""
        for prefix in self._ARCH_PREFIXES:
            field = self.gguf_reader.fields.get(f"{prefix}.{suffix}")
            if field is not None:
                return field.contents()
        return None

    def _head_dim(self) -> int:
        """head_dim, from rope.dimension_count with a head-count cross-check."""
        rope_dim = self._meta("rope.dimension_count")
        heads = self._meta("attention.head_count")
        embedding_length = self._meta("embedding_length")

        derived = None
        if heads and embedding_length:
            derived = embedding_length // heads

        if rope_dim is None:
            if derived is None:
                raise KeyError(
                    "Cannot determine head_dim: GGUF has neither "
                    "rope.dimension_count nor embedding_length/head_count"
                )
            return int(derived)

        if derived is not None and derived != rope_dim:
            # Granite applies RoPE to the full head, so these must agree. If
            # they don't, the model has a partial rotary factor and the q/k
            # permutation below would be wrong -- refuse rather than emit
            # silently broken weights.
            raise ValueError(
                f"head_dim is ambiguous: rope.dimension_count={rope_dim} but "
                f"embedding_length/head_count={derived}. Partial-rotary Granite "
                f"variants are not supported."
            )
        return int(rope_dim)

    def _fold_factors(self) -> dict[str, float]:
        """Per-tensor multiplicative folds, keyed by q4nx name substring.

        Only entries that actually change anything are returned, so a checkpoint
        with all-unit multipliers (granite-4.2-3b has three of the four at 1.0)
        does no work and touches no weights.
        """
        head_dim = self._head_dim()

        attention_multiplier = self._meta("attention.scale")
        embedding_multiplier = self._meta("embedding_scale")
        residual_multiplier = self._meta("residual_scale")
        logits_scaling = self._meta("logit_scale")

        if attention_multiplier is None:
            # A GGUF without the key is a plain Llama-scaled model; the llama
            # runtime's own hd**-0.5 is then already correct.
            attention_multiplier = head_dim ** -0.5
        # GGUF metadata comes back as numpy scalars; keep the arithmetic and the
        # `!= 1.0` guards below in plain Python floats.
        attention_multiplier = float(attention_multiplier)
        embedding_multiplier = 1.0 if embedding_multiplier is None else float(embedding_multiplier)
        residual_multiplier = 1.0 if residual_multiplier is None else float(residual_multiplier)
        logits_scaling = 1.0 if logits_scaling is None else float(logits_scaling)

        print(
            f"[INFO] Granite multipliers: attention={attention_multiplier}, "
            f"embedding={embedding_multiplier}, residual={residual_multiplier}, "
            f"logits_scaling={logits_scaling} (head_dim={head_dim})"
        )

        # The llama engine applies hd**-0.5; scale q so the product is
        # attention_multiplier.
        q_fold = attention_multiplier * math.sqrt(head_dim)

        folds: dict[str, float] = {}
        if q_fold != 1.0:
            folds["self_attn.q_proj.weight"] = q_fold
        if residual_multiplier != 1.0:
            folds["self_attn.o_proj.weight"] = residual_multiplier
            folds["mlp.down_proj.weight"] = residual_multiplier
        if embedding_multiplier != 1.0:
            folds["model.embed_tokens.weight"] = embedding_multiplier
        if logits_scaling != 1.0:
            folds["lm_head.weight"] = 1.0 / logits_scaling

        for name, factor in folds.items():
            print(f"[INFO] Granite fold: {name} *= {factor}")
        if not folds:
            print("[INFO] Granite folds: none needed (all multipliers are 1.0)")
        return folds

    @staticmethod
    def _fold_factor_for(q4nx_name: str, folds: dict[str, float]) -> float | None:
        for suffix, factor in folds.items():
            if q4nx_name.endswith(suffix):
                return factor
        return None

    def _pad_plan(self) -> tuple[int, int, float] | None:
        """`(hidden, padded_hidden, norm_scale)`, or None when not padding.

        Why this exists: `llama_npu.dll` does not accept an arbitrary hidden
        size. It whitelists one, and the accepted set is exactly the hidden
        sizes of the shipped llama-family designs -- {2048, 3072, 4096},
        measured in NpuEmbeddings tasks/0144. Granite-4.2-3B is 2560 and is
        refused outright with "Unsupported hidden size: 2560".

        Widening the residual stream to the next accepted size is exact, not an
        approximation, because every tensor that writes into the residual
        stream gets zero rows and every tensor that reads from it gets zero
        columns -- so the padded lanes are identically zero at every layer.

        The one place that is NOT automatic is RMSNorm, which divides by
        `sqrt(mean(x^2) + eps)` over the *full* width. Widening H -> H' shrinks
        the mean by H/H'. Both terms are fixed by scaling:

            norm weights *= sqrt(H / H')
            rms_norm_eps *= H / H'

        With `eps' = eps * H/H'` the denominator becomes
        `sqrt((H/H')(S/H + eps))`, so the norm weight scale `sqrt(H/H')`
        cancels it for every input rather than approximately.
        (`rms_norm_eps` is written by `model_assets.apply_granite_pad_to_config`.)

        Exact in real arithmetic, with **one** rounding in practice: the
        rescaled norm weights are stored bf16, so they carry a relative error up
        to bf16's epsilon. Measured against the unpadded build, that is
        3.5e-03 max / 1.5e-03 mean against an epsilon of 3.9e-03, and it moves
        the logits by cosine 1.5e-04 with the argmax unchanged -- an order of
        magnitude below the Q4_1 weight floor the model already sits on. The
        padded lanes themselves are exactly zero.

        The cost is arithmetic on 512 lanes of zeros -- about 20% of the hidden
        dimension -- in exchange for landing on a design that already exists.
        """
        if not self.pad_hidden:
            return None
        heads = self._meta("attention.head_count")
        hidden = int(self._meta("embedding_length"))
        target = int(self.pad_hidden)
        if target == hidden:
            return None
        if target < hidden:
            raise ValueError(
                f"--pad-hidden {target} is smaller than the model's hidden size {hidden}"
            )
        if target % 256:
            # A Q4NX tile is 32 rows x 256 K; a hidden size that is not a whole
            # number of column-blocks would need partial tiles.
            raise ValueError(f"--pad-hidden {target} is not a multiple of 256")
        if not self.pad_norm_fix:
            print("[WARN] --no-pad-norm-fix: RMSNorm's width term is NOT corrected. "
                  "On a runtime that normalizes over the padded width every "
                  "activation is scaled by sqrt(H'/H).")
            return hidden, target, 1.0
        return hidden, target, math.sqrt(hidden / target)

    @staticmethod
    def _pad_unpacked(unpacked, rows: int | None = None, cols: int | None = None):
        """Zero-extend a tensor to `rows` x `cols` (None on an axis = leave it).

        Both axes at once, because a tensor can need both: `down_proj` is
        hidden-rows by intermediate-cols, so padding hidden and intermediate
        together touches the same tensor twice.

        Q4_1 groups are 32 wide and every width involved is a multiple of 32, so
        the padding lands on whole groups: the added `d` and `m` are 0 and the
        added codes are 0, giving `w = 0*0 + 0 = 0` exactly.
        """
        if len(unpacked) == 1:
            w = unpacked[0]
            if rows is not None:
                raise ValueError("float passthrough tensors only pad on cols")
            if cols is None:
                return unpacked
            out = torch.zeros((w.shape[0], cols), dtype=w.dtype)
            out[:, : w.shape[1]] = w
            return (out,)

        d, m, qs = unpacked
        n_rows = rows if rows is not None else qs.shape[0]
        n_cols = cols if cols is not None else qs.shape[1]
        if n_rows == qs.shape[0] and n_cols == qs.shape[1]:
            return unpacked

        nd = torch.zeros((n_rows, n_cols // 32), dtype=d.dtype)
        nm = torch.zeros((n_rows, n_cols // 32), dtype=m.dtype)
        nq = torch.zeros((n_rows, n_cols), dtype=qs.dtype)
        nd[: d.shape[0], : d.shape[1]] = d
        nm[: m.shape[0], : m.shape[1]] = m
        nq[: qs.shape[0], : qs.shape[1]] = qs
        return (nd, nm, nq)

    @staticmethod
    def _endswith_any(name: str, suffixes) -> bool:
        return any(name.endswith(s) for s in suffixes)

    @staticmethod
    def _scale_unpacked(unpacked, factor: float):
        """Scale a tensor in whatever form `GGUFTensor.unpack` returned it.

        `(d, m, qs)` for a block-quantized tensor -- scale the per-block scale
        and minimum, leave the codes alone. `(w,)` for a float passthrough --
        scale the values.
        """
        if len(unpacked) == 3:
            d, m, qs = unpacked
            return (d * factor, m * factor, qs)
        if len(unpacked) == 1:
            return (unpacked[0] * factor,)
        raise ValueError(f"Unexpected unpacked tensor arity: {len(unpacked)}")

    def _convert_gguf(self, q4nx_path: str, weights_type: str):
        print("[INFO] Converting granite model to Q4NX format...")
        folds = self._fold_factors()
        head_dim = self._head_dim()
        pad = self._pad_plan()
        if pad:
            hidden, padded, norm_scale = pad
            print(f"[INFO] Granite: padding hidden {hidden} -> {padded} "
                  f"(norm weights *= {norm_scale:.9f}, rms_norm_eps *= {hidden/padded:.9f})")
        if self.pad_intermediate:
            inter = self._meta("feed_forward_length")
            print(f"[INFO] Granite: padding intermediate {inter} -> "
                  f"{int(self.pad_intermediate)} (exact; no norm spans this axis)")

        if not self._has_lm_head():
            # Tied embeddings. lm_head must come from the UNSCALED embedding
            # table, so it is built here, before the embedding fold is applied
            # in the loop below, and carries only its own logits_scaling fold.
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
            lm_head_fold = folds.get("lm_head.weight")
            if lm_head_fold is not None:
                unpacked = self._scale_unpacked(unpacked, lm_head_fold)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

        for gguf_tensor in self.gguf_tensors.values():
            if gguf_tensor.name not in self.forward_name_map:
                print(f"[WARN] Unmapped GGUF tensor, skipping: {gguf_tensor.name}")
                continue
            q4nx_name = self.forward_name_map[gguf_tensor.name]

            if "token_embd.weight" in gguf_tensor.name:
                from gguf import dequantize

                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                factor = self._fold_factor_for(q4nx_name, folds)
                if factor is not None:
                    w = (w.float() * factor).to(torch.bfloat16)
                if pad:
                    # embed_tokens is (vocab, hidden): hidden is the last axis.
                    padded_w = torch.zeros((w.shape[0], pad[1]), dtype=w.dtype)
                    padded_w[:, : w.shape[1]] = w
                    w = padded_w
                self.q4nx_tensors[q4nx_name] = w
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)

            if "q_proj" in q4nx_name or "k_proj" in q4nx_name:
                from einops import rearrange

                pp = head_dim // 2
                d, m, qw = unpacked
                d = rearrange(d, '(g p q) c -> (g q p) c', p=pp, q=2).contiguous()
                m = rearrange(m, '(g p q) c -> (g q p) c', p=pp, q=2).contiguous()
                qw = rearrange(qw, '(g p q) c -> (g q p) c', p=pp, q=2).contiguous()
                unpacked = (d, m, qw)

            factor = self._fold_factor_for(q4nx_name, folds)
            if factor is not None:
                unpacked = self._scale_unpacked(unpacked, factor)

            if pad or self.pad_intermediate:
                if pad and self._endswith_any(q4nx_name, self._NORMS):
                    # RMSNorm: widen with zero weights and rescale so the wider
                    # denominator cancels. See _pad_plan.
                    _, padded, norm_scale = pad
                    unpacked = self._scale_unpacked(unpacked, norm_scale)
                    w = unpacked[0]
                    grown = torch.zeros((padded,), dtype=w.dtype)
                    grown[: w.shape[0]] = w.reshape(-1)
                    unpacked = (grown,)
                else:
                    rows = cols = None
                    if pad:
                        padded = pad[1]
                        if self._endswith_any(q4nx_name, self._PAD_COLS):
                            cols = padded
                        elif self._endswith_any(q4nx_name, self._PAD_ROWS):
                            rows = padded
                    if self.pad_intermediate:
                        if self._endswith_any(q4nx_name, self._PAD_INTER_ROWS):
                            rows = int(self.pad_intermediate)
                        elif self._endswith_any(q4nx_name, self._PAD_INTER_COLS):
                            cols = int(self.pad_intermediate)
                    if rows is not None or cols is not None:
                        unpacked = self._pad_unpacked(unpacked, rows, cols)

            self.q4nx_tensors[q4nx_name] = self._pack_q4nx(*unpacked)

        self._export_weights(q4nx_path, weights_type)
        self._extract_tokenizer_json(q4nx_path)

    def _convert_hf(self, q4nx_path: str, weights_type: str):
        raise NotImplementedError(
            "Granite HF-safetensors conversion is not supported: the shared HF "
            "path in model_converter.py stores raw float tensors and never "
            "calls _pack_q4nx, so it does not produce a Q4NX file the FLM "
            "runtime can read. Convert from a GGUF instead "
            "(ibm-granite/granite-4.2-3b-GGUF)."
        )
