#!/usr/bin/env python3
"""xclbin_catalog.py — systematic catalog of all NPU2 xclbin kernels.

Walks an xclbin root (default: the sibling FastFlowLM checkout's
src/xclbins directory, i.e. <ModelName>/*.xclbin) and, for every kernel,
records:

  - file identity: model dir, filename, size
  - AXLF container info: UUID, section table
  - kernel interface: name + argument list (from CONNECTIVITY XML)
  - memory topology and AIE partition descriptor
  - CDO stats: tiles touched, static DM bytes, DMA-BD tiles, PM loads

Output:
  - human-readable summary on stdout
  - machine-readable JSON artifact (--json-out)
  - markdown catalog artifact   (--md-out)

Usage:
  venv/bin/python tools/xclbin_catalog.py
  venv/bin/python tools/xclbin_catalog.py --xclbin-root /path/to/src/xclbins
  FLM_XCLBIN_ROOT=/path/to/src/xclbins tools/xclbin_catalog.py --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import axlf as axlf_mod  # noqa: E402
import cdo as cdo_mod  # noqa: E402

DEFAULT_XCLBIN_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "FastFlowLM" / "src" / "xclbins"
)

SECTION_KINDS = dict(axlf_mod.SECTION_KINDS)


def default_root() -> Path:
    env = os.environ.get("FLM_XCLBIN_ROOT")
    if env:
        return Path(env).expanduser()
    return DEFAULT_XCLBIN_ROOT


# ── per-xclbin analysis ───────────────────────────────────────────────────────

def analyze_single_xclbin(path: Path) -> dict:
    """Build the catalog record for one .xclbin file."""
    result = {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "uuid": None,
        "kernel_name": None,
        "kernel_args": [],
        "sections": [],
        "mem_topology": [],
        "aie_partition": None,
        "cdo": None,
        "error": None,
    }

    try:
        xc = axlf_mod.load(path)
    except Exception as e:  # noqa: BLE001 — a bad file must not kill the scan
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["uuid"] = xc.uuid.hex() if hasattr(xc.uuid, "hex") else str(xc.uuid)
    result["kernel_name"] = xc.kernel_name
    result["kernel_args"] = [
        {"id": a.get("id"), "name": a.get("name"), "type": a.get("type"),
         "size": a.get("size"), "addressQualifier": a.get("addressQualifier")}
        for a in xc.kernel_args
    ]
    result["sections"] = [
        {"kind": s.kind,
         "name": SECTION_KINDS.get(s.kind, f"0x{s.kind:02x}"),
         "section_name": s.name,
         "offset": s.offset,
         "size": s.size}
        for s in xc.sections
    ]

    mem = xc.section(0x06)
    if mem is not None:
        topo = mem.mem_topology or []
        result["mem_topology"] = [
            {"name": t["name"], "base": t["base"], "size_kb": t["size_kb"],
             "used": t["used"]}
            for t in topo
        ]

    aie = xc.section(0x19)
    if aie is not None:
        result["aie_partition"] = aie.aie_partition_info

    pdi = xc.aie_pdi
    if pdi is not None:
        parser = cdo_mod.parse_xclbin_cdo(pdi.data)
        dm_bytes = {}
        pm_bytes = {}
        bd_tiles = set()
        tiles = set()
        for cmd in parser.cmds:
            if hasattr(cmd, "col"):
                tiles.add((cmd.col, cmd.row))
            if isinstance(cmd, cdo_mod.CdoDmaWriteCmd):
                if cmd.is_dm_init:
                    dm_bytes[(cmd.col, cmd.row)] = (
                        dm_bytes.get((cmd.col, cmd.row), 0) + len(cmd.data) * 4)
                if cmd.is_pm_load:
                    pm_bytes[(cmd.col, cmd.row)] = (
                        pm_bytes.get((cmd.col, cmd.row), 0) + len(cmd.data) * 4)
                if cmd.is_core_dma_bd or cmd.is_mem_dma_bd:
                    bd_tiles.add((cmd.col, cmd.row))
        programs = parser.extract_tile_programs()
        result["cdo"] = {
            "num_commands": len(parser.cmds),
            "tiles_touched": len(tiles),
            "dma_bd_tiles": len(bd_tiles),
            "static_dm_bytes_total": sum(dm_bytes.values()),
            "static_dm_bytes_per_tile": {
                f"({c},{r})": n for (c, r), n in sorted(dm_bytes.items())},
            "pm_load_bytes_per_tile": {
                f"({c},{r})": n for (c, r), n in sorted(pm_bytes.items())},
            "tile_programs": [p.describe() for p in programs],
        }

    return result


# ── root-directory scan ───────────────────────────────────────────────────────

def scan(root: Path, quiet: bool = False) -> list[dict]:
    """Scan <root>/<model>/<*.xclbin> and return one record per file."""
    records = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for xclbin_file in sorted(model_dir.glob("*.xclbin")):
            if not quiet:
                print(f"  analyzing {model_dir.name}/{xclbin_file.name} ...",
                      flush=True)
            rec = analyze_single_xclbin(xclbin_file)
            rec = {"model": model_dir.name, **rec}
            records.append(rec)
    return records


# ── output rendering ──────────────────────────────────────────────────────────

def render_stdout(records: list[dict]) -> str:
    lines = []
    models = sorted({r["model"] for r in records})
    lines.append(f"xclbin catalog — {len(models)} models, {len(records)} kernels")
    lines.append("=" * 78)
    current_model = None
    for r in records:
        if r["model"] != current_model:
            current_model = r["model"]
            lines.append(f"\n{current_model}")
            lines.append("-" * len(current_model))
        size_kb = r["size_bytes"] / 1024
        head = (f"  {r['filename']:<24} {size_kb:>9.1f} KB  "
                f"kernel={r['kernel_name']}")
        lines.append(head)
        if r["error"]:
            lines.append(f"    ERROR: {r['error']}")
            continue
        arg_names = ", ".join(a["name"] for a in r["kernel_args"])
        lines.append(f"    uuid={r['uuid']}  args=[{arg_names}]")
        sections = ", ".join(
            f"{s['name']}/{s['size']:,}B" for s in r["sections"])
        lines.append(f"    sections: {sections}")
        c = r["cdo"]
        if c:
            lines.append(
                f"    CDO cmds={c['num_commands']}  tiles={c['tiles_touched']}  "
                f"bd_tiles={c['dma_bd_tiles']}  "
                f"static_dm={c['static_dm_bytes_total']}B  "
                f"pm_loads={len(c['tile_programs'])}")
        elif r["aie_partition"]:
            lines.append(f"    AIE partition: {r['aie_partition']}")
    return "\n".join(lines)


def render_markdown(records: list[dict]) -> str:
    out = ["# NPU2 xclbin catalog", "",
           f"{len({r['model'] for r in records})} models, "
           f"{len(records)} kernels.", ""]
    out.append("| Model | Kernel | File | Size (KB) | UUID | Args | Tiles | Static DM |")
    out.append("|---|---|---|---:|---|---|---:|---:|")
    for r in sorted(records, key=lambda r: (r["model"], r["filename"])):
        args = ", ".join(a["name"] for a in r["kernel_args"]) or "—"
        tiles = r["cdo"]["tiles_touched"] if r["cdo"] else "—"
        dm = (f"{r['cdo']['static_dm_bytes_total']:,}" if r["cdo"] else "—")
        uuid = (r["uuid"][:8] + "…") if r["uuid"] else "—"
        out.append(
            f"| {r['model']} | {r['kernel_name'] or '—'} | {r['filename']} "
            f"| {r['size_bytes'] / 1024:.1f} | `{uuid}` | {args} | {tiles} | {dm} |")
    out.append("")
    return "\n".join(out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xclbin-root", default=None,
                    help=f"root containing <Model>/*.xclbin "
                         f"(default: {DEFAULT_XCLBIN_ROOT})")
    ap.add_argument("--json-out", metavar="PATH",
                    help="write the full catalog as JSON to PATH")
    ap.add_argument("--md-out", metavar="PATH",
                    help="write a markdown summary table to PATH")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-file progress output")
    args = ap.parse_args(argv)

    root = Path(args.xclbin_root) if args.xclbin_root else default_root()
    if not root.is_dir():
        print(f"[!] xclbin root not found: {root}", file=sys.stderr)
        print("    pass --xclbin-root or set FLM_XCLBIN_ROOT", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Scanning xclbins under {root} ...")

    records = scan(root, quiet=args.quiet)
    errors = [r for r in records if r["error"]]
    if errors:
        print(f"\n[!] {len(errors)} file(s) failed to parse:", file=sys.stderr)
        for r in errors:
            print(f"    {r['model']}/{r['filename']}: {r['error']}",
                  file=sys.stderr)

    if not args.quiet:
        print()
        print(render_stdout(records))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(records, indent=2) + "\n")
        if not args.quiet:
            print(f"\n[+] wrote {args.json_out}")

    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text(render_markdown(records))
        if not args.quiet:
            print(f"[+] wrote {args.md_out}")

    if not args.json_out and not args.md_out:
        # keep the historical behavior of writing both artifacts next to docs/
        docs = Path(__file__).resolve().parent.parent / "docs"
        Path(docs / "xclbin_catalog.json").parent.mkdir(parents=True, exist_ok=True)
        (docs / "xclbin_catalog.json").write_text(json.dumps(records, indent=2) + "\n")
        (docs / "xclbin_catalog.md").write_text(render_markdown(records))
        if not args.quiet:
            print(f"\n[+] wrote {docs / 'xclbin_catalog.json'}")
            print(f"[+] wrote {docs / 'xclbin_catalog.md'}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
