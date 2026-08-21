"""Structural and numeric sanity checks for a converted Q4NX model directory.

Usage (via the ``q4nx-build`` console script)::

    q4nx-build --verify <output-dir>

This does *not* require a reference model to compare against (see
``q4nx.tools.compare`` for that) -- it checks that the output directory is
internally consistent: required files are present, ``config.json`` has the
keys FLM expects (and lacks converter-internal leftovers), every tensor the
architecture's name-map says should exist is present with a plausible
dtype/shape, and no tensor contains NaN/Inf or other numerically-suspicious
values (e.g. the "+1 layernorm convention applied where it shouldn't be"
class of bug, which shows up as an entire tensor shifted by ~1.0).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file

from ..constants import ModelArch, ModelArchConfigs, ModelArchNames, QWEN35_VARIANT_DIMS
from ..model_converter import resolve_configs_dir


REQUIRED_BY_MODALITY = {
    "language": ["model.q4nx"],
    "vision": ["vision_weight.q4nx"],
    "audio": ["audio_weight.q4nx"],
}
ALWAYS_REQUIRED = ["config.json", "tokenizer.json", "tokenizer_config.json"]
RECOMMENDED = ["chat_template.jinja"]

# Keys that are converter-internal implementation details and must never
# leak into the FLM-facing config.json (regression checks for bugs fixed
# during HF-source conversion development).
FORBIDDEN_CONFIG_KEYS = ["vision_MM_K", "vision_MM_N", "video_token_id"]

# Norm-family substrings; the Q4NX runtime expects "weight + 1" (llama.cpp
# convention) for all of these except ssm_norm, which is stored raw.
NORM_NAME_HINTS = ("layernorm", "norm")


@dataclass
class Issue:
    level: str  # "error" | "warning"
    message: str


@dataclass
class VerifyReport:
    output_dir: Path
    issues: List[Issue] = field(default_factory=list)
    tensor_counts: Dict[str, int] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.issues.append(Issue("error", msg))

    def warn(self, msg: str) -> None:
        self.issues.append(Issue("warning", msg))

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def print_summary(self) -> None:
        errors = [i for i in self.issues if i.level == "error"]
        warnings = [i for i in self.issues if i.level == "warning"]
        print(f"\n[VERIFY] {self.output_dir}")
        for name, count in self.tensor_counts.items():
            print(f"  {name}: {count} tensors")
        if not self.issues:
            print("  No issues found.")
        for i in errors:
            print(f"  [ERROR] {i.message}")
        for i in warnings:
            print(f"  [WARN] {i.message}")
        status = "PASS" if self.ok else "FAIL"
        print(f"[VERIFY] {status} ({len(errors)} error(s), {len(warnings)} warning(s))\n")


def _detect_modality_files(output_dir: Path) -> Dict[str, str]:
    """Map modality -> filename for weight files present on disk."""
    present = {}
    for modality, filenames in REQUIRED_BY_MODALITY.items():
        for filename in filenames:
            if (output_dir / filename).is_file():
                present[modality] = filename
    return present


def _load_config_json(output_dir: Path, report: VerifyReport) -> Optional[dict]:
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        report.error("config.json missing")
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        report.error(f"config.json is not valid JSON: {e}")
        return None


def _check_files(output_dir: Path, modalities: Dict[str, str], report: VerifyReport) -> None:
    for filename in ALWAYS_REQUIRED:
        if not (output_dir / filename).is_file():
            report.error(f"required file missing: {filename}")
    for filename in RECOMMENDED:
        if not (output_dir / filename).is_file():
            report.warn(f"recommended file missing: {filename}")
    if "language" not in modalities and "vision" not in modalities and "audio" not in modalities:
        report.error("no model weight file found (model.q4nx / vision_weight.q4nx / audio_weight.q4nx)")


def _check_config_keys(config: dict, modalities: Dict[str, str], report: VerifyReport) -> None:
    for key in FORBIDDEN_CONFIG_KEYS:
        if key in config:
            report.error(f"config.json contains converter-internal key '{key}' (should have been stripped)")
        vc = config.get("vision_config")
        if isinstance(vc, dict) and key in vc:
            report.error(f"config.json vision_config contains converter-internal key '{key}'")

    has_vision_file = "vision" in modalities
    if has_vision_file and "vision_model_weight" not in config:
        report.error("vision_weight.q4nx is present but config.json is missing 'vision_model_weight'")
    if not has_vision_file and "vision_model_weight" in config:
        report.error("config.json has 'vision_model_weight' but no vision_weight.q4nx file is present")

    for key in ("architectures", "model_type"):
        if key not in config:
            report.warn(f"config.json is missing expected key '{key}'")

    if "language" in modalities:
        for key in ("num_hidden_layers", "hidden_size"):
            if key not in config:
                report.warn(f"config.json is missing expected language-model key '{key}'")


def _resolve_arch_config(config: dict) -> Optional[dict]:
    """Best-effort: find the q4nx arch config matching config.json's model_type.

    Mirrors the detection heuristics in model_converter._detect_hf_arch:
    qwen3_5/qwen3_5_text collapse to a single GGUF architecture name with the
    concrete size variant (0.8b/2b/4b/9b) distinguished only by hidden_size,
    since ModelArchNames has no per-variant model_type string to match on.
    """
    model_type = (config.get("model_type") or "").lower()
    if not model_type:
        return None
    normalized = model_type.replace("_", "").replace(".", "").replace("-", "")

    arch: Optional[ModelArch] = None
    if normalized in ("qwen35", "qwen35text", "qwen35moetext", "qwen35moe", "qwen36moe", "qwen36moetext"):
        if "moe" in normalized:
            arch = ModelArch.QWEN35MOE
        else:
            hidden_size = config.get("hidden_size", 0)
            arch = next(
                (a for a, dim in QWEN35_VARIANT_DIMS.items() if dim == hidden_size),
                ModelArch.QWEN35_4B,
            )
    else:
        for candidate_arch, names in ModelArchNames.items():
            candidate_normalized = [n.lower().replace(".", "").replace("-", "") for n in names]
            if normalized in candidate_normalized:
                arch = candidate_arch
                break

    if arch is None:
        return None
    filename = ModelArchConfigs.get(arch)
    if not filename:
        return None
    path = Path(resolve_configs_dir()) / filename
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _check_tensor_coverage(
    tensors: Dict[str, torch.Tensor],
    arch_config: dict,
    modality: str,
    report: VerifyReport,
) -> None:
    """Check that every tensor *family* the architecture declares has at least
    one matching tensor present.

    This deliberately does not assume a uniform layer count per {bid}
    pattern: many architectures have tensors that only exist at a subset of
    layers (e.g. Qwen3.5's full-attention q_proj/k_proj only exists at
    full_attention layers, not every layer; vision blocks may have a
    different depth than the language model's num_hidden_layers). Requiring
    every {bid} to resolve for every pattern produced false positives from
    exactly this kind of conditional-per-layer structure. Instead, each
    pattern is turned into a regex and checked against the actual tensor
    names: entirely absent families are flagged, partial coverage is not.
    """
    tensor_names = list(tensors.keys())
    name_map = arch_config.get("name_map", {})
    checked = 0
    for param_info in name_map.values():
        q4nx_name = param_info.get("q4nx_name", "")
        if "rope_freqs" in q4nx_name:
            continue
        is_vision = "visual" in q4nx_name
        is_audio = "audio" in q4nx_name
        if modality == "language" and (is_vision or is_audio):
            continue
        if modality == "vision" and not is_vision:
            continue
        if modality == "audio" and not is_audio:
            continue
        checked += 1
        if "{bid}" in q4nx_name:
            pattern = re.escape(q4nx_name).replace(r"\{bid\}", r"\d+")
            if not any(re.fullmatch(pattern, n) for n in tensor_names):
                report.error(f"{modality}: no tensor found matching family '{q4nx_name}' (at any layer index)")
        else:
            if q4nx_name not in tensors:
                report.error(f"{modality}: expected tensor '{q4nx_name}' is entirely missing")

    if checked == 0:
        report.warn(f"{modality}: arch config's name_map had no entries applicable to this modality")


def _check_numeric_sanity(tensors: Dict[str, torch.Tensor], report: VerifyReport) -> None:
    for name, tensor in tensors.items():
        if tensor.numel() == 0:
            report.warn(f"tensor '{name}' is empty")
            continue
        if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            t = tensor.float()
            if torch.isnan(t).any():
                report.error(f"tensor '{name}' contains NaN values")
            if torch.isinf(t).any():
                report.error(f"tensor '{name}' contains Inf values")
            if torch.all(t == 0):
                report.warn(f"tensor '{name}' is entirely zero")
            # Norm-family sanity: a *language-model* layernorm weight (which
            # should use the weight+1 convention, per llama.cpp) that goes
            # negative is a strong signal the +1 was never applied -- trained
            # RMSNorm gains are essentially always non-negative in practice,
            # so this is a much sharper signal than comparing the mean to 1.0
            # (which false-positives on raw ssm_norm weights that are also
            # naturally close to 1.0). Scoped to language-model tensors only:
            # vision-encoder norms (model.visual.*) and ssm_norm are stored
            # raw, without the +1 convention, by design.
            is_language_norm = (
                any(h in name for h in NORM_NAME_HINTS)
                and ".bias" not in name
                and "ssm_norm" not in name
                and "visual" not in name
                and "audio" not in name
            )
            if is_language_norm:
                frac_negative = (t < 0).float().mean().item()
                # A handful of near-zero elements landing just below 0 after
                # bf16 rounding is normal; a missing +1 shift instead shows
                # up as roughly half the tensor going negative (the gain
                # distribution re-centers on 0 instead of 1). Use a loose
                # threshold so single-element rounding noise doesn't trigger.
                if frac_negative > 0.10:
                    report.warn(
                        f"tensor '{name}' (layernorm, expected weight+1 convention) "
                        f"has {frac_negative:.0%} negative values -- check the +1 convention was applied"
                    )


def verify_output_dir(output_dir: Path) -> VerifyReport:
    report = VerifyReport(output_dir=output_dir)
    if not output_dir.is_dir():
        report.error(f"not a directory: {output_dir}")
        return report

    modalities = _detect_modality_files(output_dir)
    _check_files(output_dir, modalities, report)

    config = _load_config_json(output_dir, report)
    if config is not None:
        _check_config_keys(config, modalities, report)
        arch_config = _resolve_arch_config(config)
        if arch_config is None:
            report.warn(
                f"could not resolve a Q4NX arch config for model_type="
                f"'{config.get('model_type')}'; skipping tensor-coverage check"
            )
    else:
        arch_config = None

    for modality, filename in modalities.items():
        path = output_dir / filename
        try:
            tensors = load_file(str(path))
        except Exception as e:
            report.error(f"failed to load {filename}: {e}")
            continue
        report.tensor_counts[filename] = len(tensors)
        _check_numeric_sanity(tensors, report)
        if arch_config is not None and config is not None:
            _check_tensor_coverage(tensors, arch_config, modality, report)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="q4nx-build --verify",
        description="Verify a converted Q4NX model directory for structural and numeric sanity.",
    )
    parser.add_argument("output_dir", help="Path to a converted Q4NX model directory")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = verify_output_dir(Path(args.output_dir))
    report.print_summary()
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
