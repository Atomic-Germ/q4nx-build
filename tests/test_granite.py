"""Tests for the Granite (dense) family: detection, and the multiplier folds.

The folds are the whole reason this family exists as more than an alias of
llama, so they are tested as arithmetic rather than only end to end -- an
end-to-end signal cannot localise which of four multipliers went wrong, and
three of the four are 1.0 in granite-4.2-3b so a bug in them is invisible on
that checkpoint alone.
"""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.arch_detect import override_resolves_exactly  # noqa: E402
from q4nx.constants import ModelArch  # noqa: E402
from q4nx.model_converter import get_model_arch_from_gguf, get_registered_models  # noqa: E402
from q4nx.models.granite import Granite  # noqa: E402
from q4nx.models.llama import Llama  # noqa: E402

# granite-4.2-3b's real values, from the HF config.json.
G42_ATTENTION_MULTIPLIER = 0.015625
G42_HEAD_DIM = 64  # hidden_size 2560 / num_attention_heads 40


def _field(value=None, raw: str = None, name: str = ""):
    """GGUFReader field stand-in: raw string access plus `.contents()`.

    A real reader returns the string from `contents()` for a string field, and
    `model_assets._gguf_field` trusts `contents()` first. A fake that returned
    None there would silently make every arch-prefixed lookup miss.
    """
    contents_value = value if value is not None else raw
    return SimpleNamespace(
        parts=[(raw or "").encode()],
        data=[0] if raw is not None else [],
        contents=(lambda: contents_value),
        name=name,
    )


class FakeReader:
    """Just enough GGUFReader for arch detection and metadata reads."""

    def __init__(self, architecture=None, tensor_names=(), **meta):
        self.fields = {}
        if architecture:
            self.fields["general.architecture"] = _field(
                raw=architecture, name="general.architecture"
            )
        # Metadata keys are arch-prefixed in a real GGUF ("granite.attention.scale");
        # `__` in the kwarg stands for the `.` inside the suffix.
        for key, value in meta.items():
            key = f"{architecture}." + key.replace("__", ".")
            self.fields[key] = _field(value=value, name=key)
        self.tensors = [SimpleNamespace(name=n) for n in tensor_names]

    def get(self, key):
        return self.fields.get(key)


def _granite(**meta):
    """A Granite converter bound to a FakeReader, without touching a real file."""
    obj = object.__new__(Granite)
    obj.gguf_reader = FakeReader(architecture="granite", **meta)
    return obj


def _g42(**overrides):
    """granite-4.2-3b's metadata, with optional overrides."""
    meta = dict(
        rope__dimension_count=G42_HEAD_DIM,
        attention__head_count=40,
        embedding_length=2560,
        attention__scale=G42_ATTENTION_MULTIPLIER,
        embedding_scale=1.0,
        residual_scale=1.0,
        logit_scale=1.0,
    )
    meta.update(overrides)
    return _granite(**meta)


class GraniteRoutingTest(unittest.TestCase):
    def test_registered_and_subclasses_llama(self):
        self.assertIs(get_registered_models()[ModelArch.GRANITE], Granite)
        self.assertTrue(issubclass(Granite, Llama))

    def test_architecture_string_routes_to_granite_not_llama(self):
        reader = FakeReader(architecture="granite")
        self.assertEqual(get_model_arch_from_gguf(reader, ""), ModelArch.GRANITE)

    def test_override_resolves_exactly(self):
        self.assertTrue(override_resolves_exactly("granite"))


class GraniteHeadDimTest(unittest.TestCase):
    def test_rope_dimension_count(self):
        self.assertEqual(_g42()._head_dim(), G42_HEAD_DIM)

    def test_derived_from_head_count_when_rope_dim_absent(self):
        conv = _granite(attention__head_count=40, embedding_length=2560)
        self.assertEqual(conv._head_dim(), 64)

    def test_partial_rotary_is_refused_not_guessed(self):
        # rope dim 48 against a 64-wide head: the q/k permutation would be
        # wrong, so this must raise rather than emit broken weights.
        conv = _g42(rope__dimension_count=48)
        with self.assertRaises(ValueError):
            conv._head_dim()

    def test_missing_everything_raises(self):
        with self.assertRaises(KeyError):
            _granite()._head_dim()


class GraniteFoldFactorTest(unittest.TestCase):
    def test_granite_4_2_3b_folds_only_q_proj(self):
        folds = _g42()._fold_factors()
        # 0.015625 * sqrt(64) = 0.125
        self.assertEqual(list(folds), ["self_attn.q_proj.weight"])
        self.assertAlmostEqual(folds["self_attn.q_proj.weight"], 0.125)

    def test_q_fold_makes_the_effective_scale_equal_the_multiplier(self):
        # The llama engine applies hd**-0.5; after the fold the product must be
        # exactly attention_multiplier. This is the identity the fold exists for.
        folds = _g42()._fold_factors()
        effective = folds["self_attn.q_proj.weight"] * (G42_HEAD_DIM ** -0.5)
        self.assertAlmostEqual(effective, G42_ATTENTION_MULTIPLIER, places=12)

    def test_llama_scaled_model_needs_no_fold(self):
        # attention.scale absent => already hd**-0.5 => q_fold is 1.0.
        conv = _granite(rope__dimension_count=64, attention__head_count=40,
                        embedding_length=2560)
        self.assertEqual(conv._fold_factors(), {})

    def test_granite_3_style_multipliers_all_fold(self):
        # Granite 3.x ships non-unit values for all four.
        conv = _g42(embedding_scale=12.0, residual_scale=0.22, logit_scale=8.0)
        folds = conv._fold_factors()
        self.assertAlmostEqual(folds["model.embed_tokens.weight"], 12.0)
        self.assertAlmostEqual(folds["self_attn.o_proj.weight"], 0.22)
        self.assertAlmostEqual(folds["mlp.down_proj.weight"], 0.22)
        self.assertAlmostEqual(folds["lm_head.weight"], 1.0 / 8.0)

    def test_residual_fold_hits_both_block_outputs(self):
        # residual_multiplier scales the output of BOTH the attention block and
        # the MLP block; missing either leaves the model subtly wrong.
        folds = _g42(residual_scale=0.5)._fold_factors()
        self.assertIn("self_attn.o_proj.weight", folds)
        self.assertIn("mlp.down_proj.weight", folds)

    def test_fold_lookup_matches_on_suffix_across_layers(self):
        folds = {"self_attn.q_proj.weight": 0.125}
        self.assertEqual(
            Granite._fold_factor_for("model.layers.7.self_attn.q_proj.weight", folds),
            0.125,
        )
        self.assertIsNone(
            Granite._fold_factor_for("model.layers.7.self_attn.k_proj.weight", folds)
        )


class GraniteScaleUnpackedTest(unittest.TestCase):
    def test_quantized_scaling_leaves_codes_untouched(self):
        d = torch.tensor([[1.0, 2.0]])
        m = torch.tensor([[-3.0, -4.0]])
        qs = torch.tensor([[5, 6, 7, 8]], dtype=torch.int8)
        sd, sm, sq = Granite._scale_unpacked((d, m, qs), 0.125)
        torch.testing.assert_close(sd, d * 0.125)
        torch.testing.assert_close(sm, m * 0.125)
        # The codes are the whole point: scaling must not requantize.
        self.assertTrue(torch.equal(sq, qs))

    def test_scaling_dm_equals_scaling_the_dequantized_weights(self):
        # w = code*d + m, so scaling (d, m) by c scales w by c exactly.
        d = torch.tensor([[0.05]])
        m = torch.tensor([[-0.4]])
        qs = torch.tensor([[0, 3, 9, 15]], dtype=torch.int8)
        c = 0.125
        before = qs.float() * d + m
        sd, sm, sq = Granite._scale_unpacked((d, m, qs), c)
        after = sq.float() * sd + sm
        torch.testing.assert_close(after, before * c, rtol=0, atol=0)

    def test_float_passthrough_is_scaled(self):
        w = torch.tensor([1.0, -2.0])
        (scaled,) = Granite._scale_unpacked((w,), 3.0)
        torch.testing.assert_close(scaled, w * 3.0)

    def test_unexpected_arity_raises(self):
        with self.assertRaises(ValueError):
            Granite._scale_unpacked((torch.zeros(1), torch.zeros(1)), 2.0)


class GraniteHfPathTest(unittest.TestCase):
    def test_hf_conversion_refuses_rather_than_emitting_unquantized(self):
        # The shared HF path never calls _pack_q4nx, so it would write a
        # float safetensors file named model.q4nx that the runtime cannot read.
        conv = object.__new__(Granite)
        with self.assertRaises(NotImplementedError):
            conv._convert_hf("out", "language")


if __name__ == "__main__":
    unittest.main()


class GraniteConfigReconciliationTest(unittest.TestCase):
    """The skeleton config is the BASE model's, and it can describe a different
    checkpoint while every dimension still matches.

    granite-4.2-3b's GGUF points `general.base_model.0.repo_url` at
    granite-4.1-3b-base, whose config carries 12.0 / 0.22 / 10.0 / tied against
    4.2's 1.0 / 1.0 / 1.0 / untied. Nothing crashed; the emitted directory
    simply described a model it did not contain.
    """

    @staticmethod
    def _skeleton():
        # The real granite-4.1-3b-base values.
        return {
            "attention_multiplier": 0.015625,
            "embedding_multiplier": 12.0,
            "residual_multiplier": 0.22,
            "logits_scaling": 10.0,
            "tie_word_embeddings": True,
            "bos_token_id": 100257,
            "pad_token_id": 100256,
            "rms_norm_eps": 1e-05,
            "hidden_size": 2560,
        }

    @staticmethod
    def _reader(tensor_names=("output.weight",), **overrides):
        meta = dict(
            attention__scale=G42_ATTENTION_MULTIPLIER,
            embedding_scale=1.0,
            residual_scale=1.0,
            logit_scale=1.0,
            rope__dimension_count=G42_HEAD_DIM,
            attention__head_count=40,
            embedding_length=2560,
        )
        meta.update(overrides)
        reader = FakeReader(architecture="granite", tensor_names=tensor_names, **meta)
        for key, value in (("tokenizer.ggml.bos_token_id", 100283),
                           ("tokenizer.ggml.eos_token_id", 100257),
                           ("tokenizer.ggml.padding_token_id", 100257)):
            reader.fields[key] = _field(value=value, name=key)
        return reader

    def test_multipliers_describe_the_folded_weights(self):
        from q4nx.model_assets import apply_granite_fold_to_config

        cfg = self._skeleton()
        apply_granite_fold_to_config(cfg, self._reader())
        # After folding the file holds Llama-scaled weights, so the config must
        # say so: the stock head_dim**-0.5, and neutral everywhere else.
        self.assertAlmostEqual(cfg["attention_multiplier"], G42_HEAD_DIM ** -0.5)
        self.assertEqual(cfg["embedding_multiplier"], 1.0)
        self.assertEqual(cfg["residual_multiplier"], 1.0)
        self.assertEqual(cfg["logits_scaling"], 1.0)

    def test_originals_are_kept_for_audit(self):
        from q4nx.model_assets import apply_granite_fold_to_config

        cfg = self._skeleton()
        apply_granite_fold_to_config(cfg, self._reader())
        self.assertAlmostEqual(
            cfg["q4nx_folded_multipliers"]["attention_multiplier"],
            G42_ATTENTION_MULTIPLIER,
        )

    def test_tie_word_embeddings_comes_from_the_gguf_tensors(self):
        from q4nx.model_assets import apply_granite_fold_to_config

        cfg = self._skeleton()
        apply_granite_fold_to_config(cfg, self._reader(tensor_names=("output.weight",)))
        self.assertFalse(cfg["tie_word_embeddings"])

        cfg = self._skeleton()
        cfg["tie_word_embeddings"] = False
        apply_granite_fold_to_config(cfg, self._reader(tensor_names=("token_embd.weight",)))
        self.assertTrue(cfg["tie_word_embeddings"])

    def test_token_ids_are_corrected_from_the_gguf(self):
        from q4nx.model_assets import reconcile_config_with_gguf

        cfg = self._skeleton()
        reconcile_config_with_gguf(cfg, self._reader())
        self.assertEqual(cfg["bos_token_id"], 100283)
        self.assertEqual(cfg["pad_token_id"], 100257)

    def test_eos_list_containing_the_gguf_id_is_left_alone(self):
        from q4nx.model_assets import reconcile_config_with_gguf

        cfg = self._skeleton()
        cfg["eos_token_id"] = [100257, 100263]
        reconcile_config_with_gguf(cfg, self._reader())
        self.assertEqual(cfg["eos_token_id"], [100257, 100263])

    def test_float32_roundtrip_is_not_reported_as_a_mismatch(self):
        # 1e-5 through float32 reads back as 9.999999747378752e-06. Warning on
        # that trains people to ignore the warning that matters.
        from q4nx.model_assets import reconcile_config_with_gguf
        import io as _io
        from contextlib import redirect_stdout

        cfg = self._skeleton()
        reader = self._reader(attention__layer_norm_rms_epsilon=9.999999747378752e-06)
        buf = _io.StringIO()
        with redirect_stdout(buf):
            reconcile_config_with_gguf(cfg, reader)
        self.assertNotIn("rms_norm_eps", buf.getvalue())

    def test_a_real_dimension_mismatch_is_reported(self):
        from q4nx.model_assets import reconcile_config_with_gguf
        import io as _io
        from contextlib import redirect_stdout

        cfg = self._skeleton()
        cfg["hidden_size"] = 4096  # skeleton disagrees with the GGUF's 2560
        buf = _io.StringIO()
        with redirect_stdout(buf):
            reconcile_config_with_gguf(cfg, self._reader())
        out = buf.getvalue()
        self.assertIn("hidden_size", out)
        # reported, not rewritten -- padded families depend on keeping theirs
        self.assertEqual(cfg["hidden_size"], 4096)
