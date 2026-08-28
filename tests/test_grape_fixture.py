"""Regression tests against the real GRaPE-1.5 adversarial model card.

The fixture (tests/GRaPE-1.5/README.md) is a genuine card that fights
extraction on every axis: no base_model anywhere, a fictional architecture
table, wrong parameter claims, and an april-fools tag. Everything asserted
here is derived from its text with the network stubbed out.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.build_plan import (  # noqa: E402
    derive_build_plan_from_card,
    extract_param_size,
    extract_repo_ids,
    infer_weights_type,
    parse_readme_frontmatter,
)

FIXTURE = Path(__file__).parent / "GRaPE-1.5" / "README.md"


@unittest.skipUnless(FIXTURE.is_file(), "fixture not present")
class GrapeFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = FIXTURE.read_text(encoding="utf-8")
        cls.frontmatter = parse_readme_frontmatter(cls.text)
        cls.card = {
            "base_model": cls.frontmatter.get("base_model"),
            "model_name": cls.frontmatter.get("model_name"),
            "pipeline_tag": cls.frontmatter.get("pipeline_tag"),
            "tags": cls.frontmatter.get("tags") or [],
        }

    def test_frontmatter_declares_vision_and_no_base_model(self):
        self.assertEqual(self.card["pipeline_tag"], "image-text-to-text")
        self.assertIsNone(self.card["base_model"])
        self.assertIn("april-fools", self.card["tags"])

    def test_body_mining_finds_the_repo_id(self):
        ids = extract_repo_ids(self.text)
        self.assertTrue(ids, "no org/name mention found in card body")
        self.assertEqual(ids[0].lower(), "sweaterdog/grape-1.5")

    def test_size_claim_extracted_from_prose(self):
        self.assertEqual(extract_param_size(self.text), "4B")

    def test_weights_type_inference(self):
        self.assertEqual(infer_weights_type(self.card), "vision")

    def test_full_plan_offline_dead_ends_gracefully(self):
        plan = derive_build_plan_from_card(
            str(FIXTURE.parent),
            fetch=lambda r: {},          # Hub unreachable / declares nothing
            probe=lambda c: False,       # no mirrors findable
            canonicalize=lambda r: r,
        )
        self.assertTrue(plan.repo_id)
        self.assertEqual(plan.repo_id.lower().split("/")[-1], "grape-1.5")
        self.assertIsNone(plan.skeleton)  # honest dead end without network
        self.assertEqual(plan.display_name, "GRaPE-1.5")
        self.assertEqual(plan.output_name, "GRaPE-1.5-4B-NPU2")


if __name__ == "__main__":
    unittest.main()
