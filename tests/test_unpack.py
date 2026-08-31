"""Round-trip tests: `_pack_q4nx` -> `unpack_q4nx` must recover the input.

This needs no model file and no download. It is the check that makes the
tensor-by-tensor diff in `tools/verify_q4nx.py` trustworthy: if the reader is
wrong, every conversion looks wrong in the same way and the diff is worthless.

Codes must come back *exactly* -- they are integers, and any tiling, nibble
parity or half-block error shows up as a permutation rather than as noise.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.constants import ModelArch  # noqa: E402
from q4nx.unpack import (  # noqa: E402
    Q4_GROUP_SIZE,
    dequantize_q4nx,
    q4nx_tile_bytes,
    unpack_q4nx,
)
from q4nx.models.granite import Granite  # noqa: E402


def _packer(row_block_size=32, col_block_size=256, parallel_size=16):
    """A Granite converter with just the packing knobs set."""
    obj = object.__new__(Granite)
    obj.row_block_size = row_block_size
    obj.col_block_size = col_block_size
    obj.parallel_size = parallel_size
    obj.keep_block_in_2D = False
    return obj


def _random_q4(rows, cols, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = torch.rand((rows, cols // Q4_GROUP_SIZE), generator=g) * 0.05 + 0.01
    m = -torch.rand((rows, cols // Q4_GROUP_SIZE), generator=g) * 0.4
    codes = torch.randint(0, 16, (rows, cols), generator=g, dtype=torch.int8)
    # bf16 is what the container stores; round the metadata to it up front so
    # the comparison isolates layout errors from format rounding.
    d = d.to(torch.bfloat16).float()
    m = m.to(torch.bfloat16).float()
    return d, m, codes


class TileGeometryTest(unittest.TestCase):
    def test_tile_is_5120_bytes(self):
        # 512 B d + 512 B m + 4096 B nibbles, the shape every dense family uses.
        self.assertEqual(q4nx_tile_bytes(32, 256), 5120)


class RoundTripTest(unittest.TestCase):
    def _round_trip(self, rows, cols, seed=0):
        d, m, codes = _random_q4(rows, cols, seed)
        packed = _packer()._pack_q4nx(d.clone(), m.clone(), codes.clone())
        self.assertEqual(packed.numel(), rows * cols // 32 // 256 * 5120)
        return (d, m, codes), unpack_q4nx(packed, rows, cols)

    def test_codes_recover_exactly(self):
        (d, m, codes), (rd, rm, rcodes) = self._round_trip(64, 512)
        # Exact: a tiling or nibble-parity error is a permutation, not noise.
        self.assertTrue(torch.equal(rcodes, codes))
        torch.testing.assert_close(rd, d, rtol=0, atol=0)
        torch.testing.assert_close(rm, m, rtol=0, atol=0)

    def test_single_tile(self):
        (_, _, codes), (_, _, rcodes) = self._round_trip(32, 256, seed=3)
        self.assertTrue(torch.equal(rcodes, codes))

    def test_many_tiles_both_axes(self):
        # Several tiles down and across at once: catches a p/q transposition
        # that a square or single-tile case would hide.
        (_, _, codes), (_, _, rcodes) = self._round_trip(128, 1024, seed=7)
        self.assertTrue(torch.equal(rcodes, codes))

    def test_granite_projection_shapes(self):
        # The real shapes of granite-4.2-3b: hidden 2560, intermediate 8192.
        for rows, cols in ((2560, 2560), (8192, 2560), (2560, 8192)):
            with self.subTest(shape=(rows, cols)):
                (_, _, codes), (_, _, rcodes) = self._round_trip(rows, cols, seed=rows)
                self.assertTrue(torch.equal(rcodes, codes))

    def test_nibble_parity_is_even_is_low(self):
        # Rows 0 and 1 of a tile share a byte. If the parity were flipped, they
        # would swap -- the exact fault that gives coherent-looking garbage.
        d, m, codes = _random_q4(32, 256, seed=11)
        codes[0, :] = 1
        codes[1, :] = 2
        packed = _packer()._pack_q4nx(d, m, codes.clone())
        _, _, rcodes = unpack_q4nx(packed, 32, 256)
        self.assertTrue(torch.all(rcodes[0] == 1))
        self.assertTrue(torch.all(rcodes[1] == 2))

    def test_half_block_boundary_is_not_swapped(self):
        # Row 15 is the last of half-block 0 and row 16 the first of half-block
        # 1; a wrong `parallel_size` split exchanges them.
        d, m, codes = _random_q4(32, 256, seed=13)
        codes[15, :] = 5
        codes[16, :] = 10
        packed = _packer()._pack_q4nx(d, m, codes.clone())
        _, _, rcodes = unpack_q4nx(packed, 32, 256)
        self.assertTrue(torch.all(rcodes[15] == 5))
        self.assertTrue(torch.all(rcodes[16] == 10))

    def test_metadata_follows_its_own_row(self):
        # d/m are indexed kb*32 + r; a swapped index would attach the wrong
        # scale to a row while leaving every code correct.
        d, m, codes = _random_q4(32, 256, seed=17)
        d[7, :] = 0.25
        m[7, :] = -1.5
        packed = _packer()._pack_q4nx(d.clone(), m.clone(), codes)
        rd, rm, _ = unpack_q4nx(packed, 32, 256)
        torch.testing.assert_close(rd[7], torch.full((8,), 0.25))
        torch.testing.assert_close(rm[7], torch.full((8,), -1.5))


class DequantizeTest(unittest.TestCase):
    def test_dequantize_matches_w_equals_code_d_plus_m(self):
        rows, cols = 64, 512
        d, m, codes = _random_q4(rows, cols, seed=23)
        expected = codes.float() * d.repeat_interleave(Q4_GROUP_SIZE, 1) + m.repeat_interleave(
            Q4_GROUP_SIZE, 1
        )
        packed = _packer()._pack_q4nx(d.clone(), m.clone(), codes.clone())
        torch.testing.assert_close(
            dequantize_q4nx(packed, rows, cols), expected, rtol=0, atol=0
        )


class GuardTest(unittest.TestCase):
    def test_unpadded_shape_is_refused(self):
        d, m, codes = _random_q4(32, 256)
        packed = _packer()._pack_q4nx(d, m, codes)
        with self.assertRaises(ValueError):
            unpack_q4nx(packed, 30, 256)


if __name__ == "__main__":
    unittest.main()
