#!/usr/bin/env python3
"""xclbin_dm_diff.py — segment-level diff of static data memory between xclbins.

Core-tile DM (0x20000+) in an NPU2 xclbin is populated by CDO DMAWRITE
commands. This tool reconstructs the per-tile write-segment lists for two
xclbins and reports:

  - added / removed / resized segments per tile
  - pure shifts (same size, different offset — e.g. a program block grew
    and pushed later segments down)
  - byte-level content diffs within same-offset segments

Two ways to specify inputs:

  # two files
  python3 xclbin_dm_diff.py old.xclbin new.xclbin [--tile 0,2]

  # two git refs of a FastFlowLM checkout (extracts src/xclbins on the fly)
  python3 xclbin_dm_diff.py --repo ../FastFlowLM v1.0.1 v1.0.2 \\
      --model Qwen3.6-35B-A3B-NPU2 --kernel mm.xclbin [--tile 0,2]
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import axlf  # noqa: E402
import cdo  # noqa: E402


def dm_segments(xclbin_path: Path) -> dict[tuple[int, int], list[tuple[int, int, bytes]]]:
    """(col,row) -> sorted [(offset, size_bytes, data)] of DM-init writes."""
    xc = axlf.load(xclbin_path)
    pdi = xc.aie_pdi
    if pdi is None:
        raise ValueError(f"no AIE partition PDI section in {xclbin_path}")
    parser = cdo.parse_xclbin_cdo(pdi.data)
    tiles: dict[tuple[int, int], list[tuple[int, int, bytes]]] = {}
    for cmd in parser.cmds:
        if not (isinstance(cmd, cdo.CdoDmaWriteCmd) and cmd.is_dm_init):
            continue
        key = (cmd.col, cmd.row)
        raw = struct.pack(f"<{len(cmd.data)}I", *cmd.data)
        tiles.setdefault(key, []).append((cmd.local_off, len(raw), raw))
    for segs in tiles.values():
        segs.sort(key=lambda s: s[0])
    return tiles


def read_blob_from_ref(repo: Path, ref: str, member: str) -> bytes:
    """Extract one file from `git archive <ref>`."""
    with tempfile.TemporaryDirectory(prefix="dm_diff_") as tmp:
        arc = Path(tmp) / "a.tar"
        with open(arc, "wb") as f:
            subprocess.run(["git", "-C", str(repo), "archive", ref,
                            "--", member], stdout=f, check=True)
        with tarfile.open(arc) as tf:
            m = tf.getmember(member)
            return tf.extractfile(m).read()


def classify(old_segs, new_segs):
    """Yield human-readable deltas between two segment lists."""
    old_map = {off: (size, data) for off, size, data in old_segs}
    new_map = {off: (size, data) for off, size, data in new_segs}

    # same-offset segments: report when size or content changed
    content = []
    for off in sorted(set(old_map) & set(new_map)):
        if old_map[off] != new_map[off]:
            content.append((off, old_map[off][1], new_map[off][1]))

    removed = [o for o in old_map if o not in new_map]
    added = [n for n in new_map if n not in old_map]

    # match removed -> added by identical size (pure shift)
    shifts, gone, fresh = [], [], []
    used = set()
    for ro in sorted(removed):
        rsize = old_map[ro][0]
        match = next((no for no in sorted(added)
                      if no not in used and new_map[no][0] == rsize), None)
        if match is not None:
            used.add(match)
            shifts.append((ro, match, rsize))
            if new_map[match][1] != old_map[ro][1]:
                content.append((match, old_map[ro][1], new_map[match][1]))
        else:
            gone.append(ro)
    fresh = [no for no in sorted(added) if no not in used]
    for ro in gone:
        content.insert(0, (None, old_map[ro][1], None))
    for no in fresh:
        content.append((no, None, new_map[no][1]))

    return sorted(shifts), sorted(content, key=lambda c: (c[0] is None, c[0] or 0))


def hex_head(data: bytes, n: int = 16) -> str:
    return data[:n].hex() + ("…" if len(data) > n else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", help="old .xclbin OR old git ref (with --repo)")
    ap.add_argument("new", help="new .xclbin OR new git ref (with --repo)")
    ap.add_argument("--repo", default=None,
                    help="FastFlowLM checkout; treats OLD/NEW as git refs")
    ap.add_argument("--model", default=None,
                    help="model dir under src/xclbins (ref mode)")
    ap.add_argument("--kernel", default=None,
                    help="kernel filename, e.g. mm.xclbin (ref mode)")
    ap.add_argument("--tile", default=None,
                    help="restrict output to one tile, e.g. '0,2'")
    args = ap.parse_args(argv)

    if args.repo:
        if not (args.model and args.kernel):
            ap.error("--repo mode needs --model and --kernel")
        member_old = f"src/xclbins/{args.model}/{args.kernel}"
        blob_a = read_blob_from_ref(Path(args.repo), args.old, member_old)
        blob_b = read_blob_from_ref(Path(args.repo), args.new, member_old)
        label = f"{member_old}: {args.old} vs {args.new}"
        with tempfile.NamedTemporaryFile(suffix=".xclbin") as fa, \
                tempfile.NamedTemporaryFile(suffix=".xclbin") as fb:
            fa.write(blob_a); fa.flush()
            fb.write(blob_b); fb.flush()
            tiles_a = dm_segments(Path(fa.name))
            tiles_b = dm_segments(Path(fb.name))
    else:
        label = f"{args.old} vs {args.new}"
        tiles_a = dm_segments(Path(args.old))
        tiles_b = dm_segments(Path(args.new))

    tile_filter = None
    if args.tile:
        c, r = (int(x) for x in args.tile.split(","))
        tile_filter = (c, r)

    print(f"── DM segment diff: {label} ──")
    keys = sorted(set(tiles_a) | set(tiles_b))
    if tile_filter:
        keys = [k for k in keys if k == tile_filter]
    total_delta = 0
    for key in keys:
        segs_a = tiles_a.get(key, [])
        segs_b = tiles_b.get(key, [])
        bytes_a = sum(s for _, s, _ in segs_a)
        bytes_b = sum(s for _, s, _ in segs_b)
        total_delta += bytes_b - bytes_a
        if segs_a == segs_b:
            continue
        print(f"\ntile{key}: {len(segs_a)} segs/{bytes_a} B -> "
              f"{len(segs_b)} segs/{bytes_b} B ({bytes_b - bytes_a:+d})")
        shifts, content = classify(segs_a, segs_b)
        for ro, no, size in shifts:
            note = "" if ro == no else f"  (shift 0x{ro:x} -> 0x{no:x})"
            marker = " =" if ro == no else " ≡"
            print(f"  {marker} {size:>6} B{note}")
        for off, old_d, new_d in content:
            if off is None:
                print(f"  - removed {len(old_d)} B @ see shift list "
                      f"[{hex_head(old_d)}]")
            elif old_d is None:
                print(f"  + added   {len(new_d)} B @ 0x{off:05x} "
                      f"[{hex_head(new_d)}]")
            else:
                ndiff = sum(1 for i in range(min(len(old_d), len(new_d)))
                            if old_d[i] != new_d[i])
                print(f"  ~ {len(old_d)} -> {len(new_d)} B @ 0x{off:05x}  "
                      f"({ndiff} byte diffs) [{hex_head(new_d)}]")

    print(f"\ntotal DM delta: {total_delta:+d} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
