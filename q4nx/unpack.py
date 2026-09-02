"""Read a Q4NX tensor back into weights -- the inverse of `_pack_q4nx`.

The converter only ever writes. Without a reader, the only check available on
its output is whether the model talks, and an end-to-end signal cannot localise
which tensor went wrong -- or even reliably detect a fault, when two faults are
present and partially cancel. This module closes that loop so a conversion can
be diffed tensor by tensor against the weights it was built from.

Layout, for `keep_block_in_2D = false` (every dense family: llama, granite, ...)
with the standard `q4nx_config` of row_block_size 32, col_block_size 256,
parallel_size 16 and a Q4 group size of 32. One tile covers 32 output rows x 256
input columns and occupies 5120 bytes:

    [ 512 B  d  as 256 bf16 ][ 512 B  m  as 256 bf16 ][ 4096 B packed nibbles ]

`d` and `m` are indexed `kb * 32 + r` for group `kb` in 0..7 and row `r` in
0..31. The nibbles are stored `(g, c, r, b)` with `g` in 0..1 (two half-blocks
of 16 rows), `c` the column within the tile, and the row within the half-block
split as `r * 2 + b` -- so **b = 0 is the low nibble and is the even row**.
Weights are Q4_1 semantics throughout: `w = code * d + m`, codes 0..15.

Derived by inverting `model_converter._pack_q4nx` directly, and cross-checked
against an independent reading of the same container in
`LLMNpuTest/tools/q4nx.py`, which was solved against ground truth rather than
from this source. The two agree on tile size, metadata indexing and nibble
parity.
"""

from __future__ import annotations

import torch
from einops import rearrange

Q4_GROUP_SIZE = 32
NUM_INT4_IN_BYTE = 2


def q4nx_tile_bytes(row_block_size: int = 32, col_block_size: int = 256) -> int:
    """Bytes per Q4NX tile: two bf16 metadata planes plus the packed nibbles."""
    meta = (col_block_size // Q4_GROUP_SIZE) * row_block_size * 2  # bf16
    codes = row_block_size * col_block_size // NUM_INT4_IN_BYTE
    return 2 * meta + codes


def unpack_q4nx(
    packed: torch.Tensor,
    rows: int,
    cols: int,
    row_block_size: int = 32,
    col_block_size: int = 256,
    parallel_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert `_pack_q4nx`, returning `(d, m, codes)` in unpacked layout.

    `d` and `m` are `(rows, cols // 32)` float32; `codes` is `(rows, cols)` int8
    holding values 0..15. `rows`/`cols` are the *padded* dimensions -- the
    packer pads up to the block sizes, and the padding is returned as-is so a
    caller can slice it off with the true shape.
    """
    if rows % row_block_size or cols % col_block_size:
        raise ValueError(
            f"({rows}, {cols}) is not a whole number of {row_block_size}x"
            f"{col_block_size} tiles; pass the padded shape"
        )

    n_tiles = (rows // row_block_size) * (cols // col_block_size)
    tile_bytes = q4nx_tile_bytes(row_block_size, col_block_size)
    packed = packed.reshape(n_tiles, tile_bytes)

    meta_bytes = (col_block_size // Q4_GROUP_SIZE) * row_block_size * 2
    d = packed[:, :meta_bytes].contiguous().view(torch.bfloat16).float()
    m = packed[:, meta_bytes : 2 * meta_bytes].contiguous().view(torch.bfloat16).float()
    codes = packed[:, 2 * meta_bytes :].contiguous().view(torch.uint8)

    # (n, (c r)) -> (rows, cols/32), undoing '(p r) (q c) -> (p q) (c r)'
    unmerge = dict(
        p=rows // row_block_size,
        r=row_block_size,
        c=col_block_size // Q4_GROUP_SIZE,
    )
    d = rearrange(d, "(p q) (c r) -> (p r) (q c)", **unmerge).contiguous()
    m = rearrange(m, "(p q) (c r) -> (p r) (q c)", **unmerge).contiguous()

    # Split each byte back into its two nibbles. b=0 is the low nibble.
    groups = row_block_size // parallel_size
    codes = rearrange(
        codes, "n (g c r) -> n g c r", g=groups, c=col_block_size,
        r=parallel_size // NUM_INT4_IN_BYTE,
    )
    low = codes & 0x0F
    high = (codes >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1)  # (n, g, c, r, b)
    codes = rearrange(codes, "n g c r b -> n (g r b) c")
    codes = rearrange(
        codes, "(p q) r c -> (p r) (q c)", p=rows // row_block_size
    ).contiguous()

    return d, m, codes.to(torch.int8)


def dequantize_q4nx(packed: torch.Tensor, rows: int, cols: int, **kwargs) -> torch.Tensor:
    """Q4NX tile stream -> `(rows, cols)` float32 weights, `w = code * d + m`."""
    d, m, codes = unpack_q4nx(packed, rows, cols, **kwargs)
    d = d.repeat_interleave(Q4_GROUP_SIZE, dim=1)
    m = m.repeat_interleave(Q4_GROUP_SIZE, dim=1)
    return codes.float() * d + m
