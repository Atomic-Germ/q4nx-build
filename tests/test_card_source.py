"""Tests for local model-card (README.md) extraction feeding -i derivation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.build_plan import (  # noqa: E402
    BuildPlan,
    derive_build_plan_from_card,
    display_name_for_arch,
    extract_param_size,
    extract_repo_ids,
    find_skeleton_for_arch,
    parse_readme_frontmatter,
)
from q4nx.constants import ModelArch  # noqa: E402

GRAPE_README = """---
license: apache-2.0
language:
  - en
  - zh
tags:
  - reasoning
  - vision
pipeline_tag: image-text-to-text
---

**GRaPE 1.5**

Some blurb about the 4-billion-parameter model.

```python
model = AutoModelForCausalLM.from_pretrained(
    "sweaterdog/GRaPE-1.5",
)
tokenizer = AutoTokenizer.from_pretrained("sweaterdog/GRaPE-1.5")
```

```bash
vllm serve sweaterdog/GRaPE-1.5 --dtype bfloat16
ollama run sweaterdog/grape-1.5
```

![chart](https://cdn-uploads.huggingface.co/production/uploads/x/img.png)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
"""


class ParseFrontmatterTest(unittest.TestCase):
    def test_flat_keys_and_lists(self):
        fm = parse_readme_frontmatter(GRAPE_README)
        self.assertEqual(fm["license"], "apache-2.0")
        self.assertEqual(fm["pipeline_tag"], "image-text-to-text")
        self.assertEqual(fm["tags"], ["reasoning", "vision"])
        self.assertEqual(fm["language"], ["en", "zh"])

    def test_no_frontmatter(self):
        self.assertEqual(parse_readme_frontmatter("just text"), {})
        self.assertEqual(parse_readme_frontmatter(""), {})

    def test_quoted_values(self):
        fm = parse_readme_frontmatter("---\nmodel_name: \"My Model\"\n---\nbody")
        self.assertEqual(fm["model_name"], "My Model")


class ExtractRepoIdsTest(unittest.TestCase):
    def test_code_mentions_rank_first(self):
        ids = extract_repo_ids(GRAPE_README)
        self.assertTrue(ids)
        self.assertEqual(ids[0].lower(), "sweaterdog/grape-1.5")

    def test_excludes_badge_and_cdn_hosts(self):
        ids = extract_repo_ids(GRAPE_README)
        lowered = [i.lower() for i in ids]
        self.assertNotIn("opensource.org/apache-2.0".lower(), lowered)
        self.assertFalse(any("cdn" in i or "shields" in i for i in lowered))

    def test_no_mentions(self):
        self.assertEqual(extract_repo_ids("nothing here"), [])


class ExtractParamSizeTest(unittest.TestCase):
    def test_badge_style(self):
        self.assertEqual(extract_param_size("Parameters-4B"), "4B")

    def test_prose_style(self):
        self.assertEqual(extract_param_size("a 4-billion-parameter multimodal model"), "4B")
        self.assertEqual(extract_param_size("0.8B params model card"), "0.8B")

    def test_absent(self):
        self.assertIsNone(extract_param_size("no size claims here"))


def _plan(tmp_path, body=GRAPE_README, fetch=None, probe=None, canonicalize=None,
          dirname="GRaPE-1.5"):
    d = tmp_path / dirname
    d.mkdir(exist_ok=True)
    (d / "README.md").write_text(body)
    kwargs = {}
    if fetch is not None:
        kwargs["fetch"] = fetch
    if probe is not None:
        kwargs["probe"] = probe
    if canonicalize is not None:
        kwargs["canonicalize"] = canonicalize
    return derive_build_plan_from_card(str(d), **kwargs)


class DeriveFromCardTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _offline_stubs(self, bases=None):
        """Stubs that simulate an unreachable-but-known Hub."""
        cards = {
            "sweaterdog/GRaPE-1.5": {"base_model": bases},
            "Qwen/Qwen3.5-4B": {"base_model": ["Qwen/Qwen3.5-4B-Base"]},
        }

        return (
            (lambda r: cards.get(r, {})),
            {"Atomic-Germ/Qwen3.5-4B-NPU2"}.__contains__,
            lambda r: r,  # identity canonicalization
        )

    def test_full_extraction_offline(self):
        fetch, probe, canon = self._offline_stubs(bases=["Qwen/Qwen3.5-4B"])
        plan = _plan(self.tmp, fetch=fetch, probe=probe, canonicalize=canon)
        self.assertIsInstance(plan, BuildPlan)
        self.assertEqual(plan.repo_id, "sweaterdog/GRaPE-1.5")
        self.assertEqual(plan.weights_type, "vision")  # pipeline_tag image-text-to-text
        self.assertEqual(plan.display_name, "GRaPE-1.5")
        # skeleton found through the locally-seeded base chain
        self.assertEqual(plan.skeleton, "Atomic-Germ/Qwen3.5-4B-NPU2")
        self.assertEqual(plan.size_token, "4B")  # from prose/badge
        # "4B" is not a segment of display "GRaPE-1.5", so it gets appended.
        self.assertEqual(plan.output_name, "GRaPE-1.5-4B-NPU2")

    def test_dead_end_chain_without_base_model(self):
        fetch, probe, canon = self._offline_stubs(bases=None)
        plan = _plan(self.tmp, fetch=fetch, probe=probe, canonicalize=canon)
        self.assertEqual(plan.chain, ["sweaterdog/GRaPE-1.5"])
        self.assertIsNone(plan.skeleton)
        self.assertEqual(plan.weights_type, "vision")
        self.assertEqual(plan.output_name, "GRaPE-1.5-4B-NPU2")

    def test_canonicalization_applies(self):
        fetch, probe, canon = self._offline_stubs()
        plan = _plan(
            self.tmp,
            fetch=fetch,
            probe=probe,
            canonicalize=lambda r: "Sweaterdog/GRaPE-1.5" if r.lower().startswith("sweaterdog/") else r,
        )
        self.assertEqual(plan.repo_id, "Sweaterdog/GRaPE-1.5")

    def test_card_without_any_repo_mention_yields_empty_plan(self):
        d = self.tmp / "Mystery"
        d.mkdir()
        (d / "README.md").write_text("---\npipeline_tag: text-generation\n---\nvague card")
        plan = derive_build_plan_from_card(
            str(d),
            fetch=lambda r: {},
            probe=lambda c: False,
            canonicalize=lambda r: None,
        )
        self.assertEqual(plan.repo_id, "")
        self.assertEqual(plan.weights_type, "language")  # CLI turns this into sys.exit


class ArchSkeletonRescueTest(unittest.TestCase):
    def test_display_names_for_sized_archs(self):
        self.assertEqual(display_name_for_arch(ModelArch.QWEN35_4B), "Qwen3.5-4B")
        self.assertEqual(display_name_for_arch(ModelArch.QWEN35_08B), "Qwen3.5-0.8B")

    def test_unsized_archs_refuse_to_guess(self):
        self.assertIsNone(display_name_for_arch(ModelArch.QWEN3))
        self.assertIsNone(display_name_for_arch(ModelArch.GEMMA4))
        self.assertIsNone(display_name_for_arch(ModelArch.QWEN35MOE))

    def test_find_skeleton_for_arch_probes_orgs_in_order(self):
        calls = []
        found = find_skeleton_for_arch(
            ModelArch.QWEN35_2B,
            probe=lambda c: (calls.append(c) or c.startswith("FastFlowLM/")),
        )
        self.assertEqual(found, "FastFlowLM/Qwen3.5-2B-NPU2")
        self.assertEqual(calls[0], "Atomic-Germ/Qwen3.5-2B-NPU2")

    def test_no_mirror_returns_none(self):
        self.assertIsNone(
            find_skeleton_for_arch(ModelArch.QWEN35_9B, probe=lambda c: False)
        )


if __name__ == "__main__":
    unittest.main()
