#!/usr/bin/env python3
"""flm_triage.py — decide whether a FastFlowLM update is small or tool-breaking.

Compares two FastFlowLM git refs (e.g. tags v1.0.1 .. v1.0.2) across the
surfaces that q4nx-build depends on, and emits a verdict:

  OK            cosmetic/docs only
  REVIEW        runtime/model code changed (weight mappings may have moved)
  KERNELS       xclbin binaries changed structurally (interface / CDO / DM)
  FORMAT        command encoding or Q4NX container headers changed
                -> txn/cdo/converter tooling must be re-derived

Surfaces inspected
──────────────────
  FORMAT   src/include/npu_utils/**            (XAIE transaction commands)
           src/include/tensor_utils/**         (Q4NX / SafeTensors container)
  KERNELS  src/xclbins/**/*.xclbin             (parsed with tools/axlf+cdo;
           only files whose sha256 changed are structurally analyzed)
  REVIEW   src/common/AutoModel/**, src/common/AutoEmbeddingModel/**
           src/include/lm_config.hpp, src/model_list.json

Usage:
  venv/bin/python tools/flm_triage.py v1.0.1 v1.0.2
  venv/bin/python tools/flm_triage.py v1.0.1 v1.0.2 --repo /path/to/FastFlowLM
  venv/bin/python tools/flm_triage.py v1.0.1 v1.0.2 --model Qwen3.6-35B-A3B-NPU2
  venv/bin/python tools/flm_triage.py v1.0.1 v1.0.2 --json

Exit codes: 0=OK 1=REVIEW 2=KERNELS 3=FORMAT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

VERDICT_ORDER = {"OK": 0, "REVIEW": 1, "KERNELS": 2, "FORMAT": 3}

# git pathspecs per surface
FORMAT_PATHS = ["src/include/npu_utils", "src/include/tensor_utils"]
REVIEW_PATHS = [
    "src/common/AutoModel",
    "src/common/AutoEmbeddingModel",
    "src/include/lm_config.hpp",
    "src/model_list.json",
]
XCLBIN_PREFIX = "src/xclbins"


def run_git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def ref_exists(repo: Path, ref: str) -> bool:
    try:
        run_git(repo, "rev-parse", "--verify", "--quiet", ref + "^{commit}")
        return True
    except subprocess.CalledProcessError:
        return False


def changed_files(repo: Path, old: str, new: str, *pathspecs: str) -> list[str]:
    out = run_git(repo, "diff", "--name-only", "--no-renames", old, new,
                  "--", *pathspecs) if pathspecs else \
        run_git(repo, "diff", "--name-only", "--no-renames", old, new)
    return [l for l in out.splitlines() if l.strip()]


# ── xclbin structural comparison ─────────────────────────────────────────────

def blob_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def read_xclbins_from_tag(repo: Path, ref: str,
                          model_filter: str | None) -> dict[str, bytes]:
    """Extract src/xclbins/** from a git ref into {relative_path: bytes}."""
    result: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="flm_triage_") as tmp:
        arc = Path(tmp) / "arc.tar"
        subprocess.run(
            ["git", "-C", str(repo), "archive", ref, "--", XCLBIN_PREFIX],
            stdout=open(arc, "wb"), check=True)
        with tarfile.open(arc) as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(".xclbin"):
                    continue
                rel = m.name[len(XCLBIN_PREFIX) + 1:]
                if model_filter and not rel.startswith(model_filter):
                    continue
                result[rel] = tf.extractfile(m).read()
    return result


def diff_xclbin_records(a: dict, b: dict) -> list[str]:
    """Human-readable structural deltas between two catalog records."""
    notes = []
    if a["kernel_name"] != b["kernel_name"]:
        notes.append(f"kernel {a['kernel_name']} -> {b['kernel_name']}")
    if [x["name"] for x in a["kernel_args"]] != [x["name"] for x in b["kernel_args"]]:
        notes.append("kernel args changed")
    sa = {s["kind"]: s["size"] for s in a["sections"]}
    sb = {s["kind"]: s["size"] for s in b["sections"]}
    if set(sa) != set(sb):
        notes.append(f"section kinds changed: "
                     f"+{sorted(set(sb) - set(sa))} -{sorted(set(sa) - set(sb))}")
    elif any(sb[k] != sa[k] for k in sa):
        big = [f"{s['name']}{sa[s['kind']]:+d}->{sb[s['kind']]}"
               for s in b["sections"] if sb[s["kind"]] != sa[s["kind"]]]
        notes.append("section sizes: " + ", ".join(big[:4]))
    ca, cb = a["cdo"], b["cdo"]
    if ca and cb:
        if ca["num_commands"] != cb["num_commands"]:
            notes.append(f"CDO cmds {ca['num_commands']} -> {cb['num_commands']}")
        if ca["tiles_touched"] != cb["tiles_touched"]:
            notes.append(f"tiles {ca['tiles_touched']} -> {cb['tiles_touched']}")
        if ca["static_dm_bytes_total"] != cb["static_dm_bytes_total"]:
            notes.append(f"static DM {ca['static_dm_bytes_total']}B -> "
                         f"{cb['static_dm_bytes_total']}B")
        ka = set(ca["static_dm_bytes_per_tile"])
        kb = set(cb["static_dm_bytes_per_tile"])
        if ka != kb:
            notes.append("DM tile map changed")
        elif any(ca["static_dm_bytes_per_tile"][k] !=
                 cb["static_dm_bytes_per_tile"][k] for k in ka):
            notes.append("per-tile DM sizes changed")
    elif (ca is None) != (cb is None):
        notes.append("CDO presence changed")
    return notes


def triage_xclbins(repo: Path, old: str, new: str,
                   model_filter: str | None, verbose: bool
                   ) -> tuple[list[str], list[dict], bool]:
    """Compare xclbin sets of two refs.

    Returns (findings, details, has_changes).
    """
    import xclbin_catalog as cat

    old_files = read_xclbins_from_tag(repo, old, model_filter)
    new_files = read_xclbins_from_tag(repo, new, model_filter)

    findings: list[str] = []
    details: list[dict] = []

    added = sorted(set(new_files) - set(old_files))
    removed = sorted(set(old_files) - set(new_files))
    for rel in added:
        findings.append(f"KERNELS  + {rel} (new)")
    for rel in removed:
        findings.append(f"KERNELS  - {rel} (removed)")

    common = sorted(set(old_files) & set(new_files))
    changed = [rel for rel in common
               if blob_sha(old_files[rel]) != blob_sha(new_files[rel])]
    identical = len(common) - len(changed)
    if verbose or changed or added or removed:
        findings.append(f"KERNELS  {identical}/{len(common)} shared xclbins "
                        f"byte-identical, {len(changed)} changed")

    # Structural analysis only for changed kernels (parse cost is real).
    cache: dict[tuple[str, str], dict] = {}

    def record(ref: str, rel: str, blob: bytes) -> dict:
        key = (ref, rel)
        if key not in cache:
            with tempfile.NamedTemporaryFile(suffix=".xclbin") as f:
                f.write(blob)
                f.flush()
                rec = cat.analyze_single_xclbin(Path(f.name))
            rec["filename"] = rel
            cache[key] = rec
        return cache[key]

    for rel in changed:
        ra = record(old, rel, old_files[rel])
        rb = record(new, rel, new_files[rel])
        notes = diff_xclbin_records(ra, rb)
        size_a, size_b = len(old_files[rel]), len(new_files[rel])
        delta = size_b - size_a
        head = (f"KERNELS  ~ {rel}  ({size_a:,} -> {size_b:,} B, "
                f"{delta:+,})")
        if notes:
            findings.append(head + "\n           " + "; ".join(notes))
        else:
            findings.append(head + "  [structure unchanged]")
        details.append({"path": rel, "size_old": size_a, "size_new": size_b,
                        "notes": notes})

    return findings, details, bool(added or removed or changed)


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_ref", help="e.g. v1.0.1")
    ap.add_argument("new_ref", help="e.g. v1.0.2")
    ap.add_argument("--repo", default=None,
                    help="FastFlowLM checkout (default: sibling ../FastFlowLM)")
    ap.add_argument("--model", default=None,
                    help="restrict xclbin analysis to one model dir, "
                         "e.g. 'GPT-OSS-20B-NPU2'")
    ap.add_argument("--skip-xclbins", action="store_true",
                    help="source-level diff only (fast)")
    ap.add_argument("--json", action="store_true", help="JSON output")
    args = ap.parse_args(argv)

    repo = Path(args.repo) if args.repo else \
        Path(__file__).resolve().parent.parent.parent / "FastFlowLM"
    if not (repo / ".git").exists():
        print(f"[!] not a git repo: {repo}", file=sys.stderr)
        return 2
    for ref in (args.old_ref, args.new_ref):
        if not ref_exists(repo, ref):
            print(f"[!] unknown ref in {repo}: {ref} "
                  f"(try: git -C {repo} fetch --tags)", file=sys.stderr)
            return 2

    verdict = "OK"
    sections: dict[str, list] = {
        "format": [], "kernels": [], "review": [], "other": []}

    # ── FORMAT surfaces ──────────────────────────────────────────────
    fmt_files = changed_files(repo, args.old_ref, args.new_ref, *FORMAT_PATHS)
    for f in fmt_files:
        sections["format"].append(f)
        print(f"FORMAT   {f}")
    if fmt_files:
        verdict = "FORMAT"

    # ── KERNELS surface ──────────────────────────────────────────────
    if not args.skip_xclbins:
        k_findings, k_details, k_changed = triage_xclbins(
            repo, args.old_ref, args.new_ref, args.model,
            verbose=args.json)
        for line in k_findings:
            print(line)
        sections["kernels"] = k_details
        if k_changed and VERDICT_ORDER[verdict] < VERDICT_ORDER["KERNELS"]:
            verdict = "KERNELS"

    # ── REVIEW surfaces ──────────────────────────────────────────────
    rev_files = changed_files(repo, args.old_ref, args.new_ref, *REVIEW_PATHS)
    for f in rev_files:
        sections["review"].append(f)
        print(f"REVIEW   {f}")
    if rev_files and VERDICT_ORDER[verdict] < VERDICT_ORDER["REVIEW"]:
        verdict = "REVIEW"

    # everything else, condensed
    all_changed = changed_files(repo, args.old_ref, args.new_ref)
    known = set(fmt_files) | set(rev_files)
    other = [f for f in all_changed
             if f not in known and not f.startswith(XCLBIN_PREFIX)]
    sections["other"] = other
    if other and not args.json:
        print(f"\nOTHER    {len(other)} unrelated files changed "
              f"(docs, ci, server, ...):")
        for f in other[:10]:
            print(f"           {f}")
        if len(other) > 10:
            print(f"           ... and {len(other) - 10} more")

    # ── verdict ──────────────────────────────────────────────────────
    advice = {
        "OK":     "nothing q4nx-build cares about changed",
        "REVIEW": "read the diffs above; weight mappings/config keys may "
                  "have moved — rerun verify_moe_q4nx against a converted model",
        "KERNELS": "AMD rebuilt kernels; compare CDO/DM structure above. "
                   "If interfaces match, conversions likely still load — "
                   "test with flm before touching the converter",
        "FORMAT": "command/Q4NX encoding changed — re-derive txn.py/cdo.py/"
                  "pack layouts from the new headers before converting",
    }[verdict]

    if args.json:
        print(json.dumps({
            "old": args.old_ref, "new": args.new_ref,
            "verdict": verdict, "advice": advice,
            **sections,
        }, indent=2))
    else:
        print("\n" + "=" * 70)
        print(f"VERDICT: {verdict} — {advice}")

    return VERDICT_ORDER[verdict]


if __name__ == "__main__":
    sys.exit(main())
