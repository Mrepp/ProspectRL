"""Tests for world simulation (Phase 1)."""

from __future__ import annotations

import time

import numpy as np
from prospect_rl.config import (
    OBS_WINDOW_RADIUS_XZ,
    OBS_WINDOW_X,
    OBS_WINDOW_Y,
    OBS_WINDOW_Y_ABOVE,
    OBS_WINDOW_Y_BELOW,
    OBS_WINDOW_Z,
    BiomeType,
    BlockType,
)
from prospect_rl.env.world.biome import BiomeGenerator
from prospect_rl.env.world.world import World


class TestWorldInit:
    """World initialisation and basic properties."""

    def test_correct_dimensions(self) -> None:
        w = World(size=(16, 16, 16), seed=1, caves_enabled=False)
        assert w.shape == (16, 16, 16)

    def test_blocks_are_int8(self) -> None:
        w = World(size=(8, 8, 8), seed=1, caves_enabled=False)
        assert w._blocks.dtype == np.int8

    def test_bedrock_at_y0(self) -> None:
        w = World(
            size=(16, 16, 16), seed=1, caves_enabled=True,
        )
        for x in range(16):
            for z in range(16):
                assert w.get_block(x, 0, z) == BlockType.BEDROCK

    def test_no_bedrock_above_y0(self) -> None:
        w = World(
            size=(16, 16, 16), seed=1, caves_enabled=False,
        )
        for x in range(16):
            for z in range(16):
                for y in range(1, 16):
                    assert w.get_block(x, y, z) != BlockType.BEDROCK


class TestOreDistribution:
    """Ore placement tests."""

    def test_ores_present(self) -> None:
        w = World(
            size=(32, 32, 32), seed=42,
            ore_density_multiplier=10.0, caves_enabled=False,
        )
        ore_count = 0
        for bt in [
            BlockType.COAL_ORE, BlockType.IRON_ORE,
            BlockType.GOLD_ORE, BlockType.DIAMOND_ORE,
            BlockType.REDSTONE_ORE, BlockType.EMERALD_ORE,
            BlockType.LAPIS_ORE, BlockType.COPPER_ORE,
        ]:
            ore_count += int(np.sum(w._blocks == bt))
        assert ore_count > 0, "No ores found in dense world"

    def test_ores_not_in_bedrock(self) -> None:
        w = World(
            size=(16, 16, 16), seed=42,
            ore_density_multiplier=10.0, caves_enabled=False,
        )
        bedrock_layer = w._blocks[:, 0, :]
        for bt in [
            BlockType.COAL_ORE, BlockType.IRON_ORE,
            BlockType.GOLD_ORE, BlockType.DIAMOND_ORE,
            BlockType.COPPER_ORE,
        ]:
            assert np.sum(bedrock_layer == bt) == 0

    def test_density_multiplier_increases_ores(self) -> None:
        w_sparse = World(
            size=(32, 32, 32), seed=42,
            ore_density_multiplier=1.0, caves_enabled=False,
        )
        w_dense = World(
            size=(32, 32, 32), seed=42,
            ore_density_multiplier=10.0, caves_enabled=False,
        )
        sparse_count = np.sum(
            (w_sparse._blocks >= BlockType.COAL_ORE)
            & (w_sparse._blocks <= BlockType.COPPER_ORE)
        )
        dense_count = np.sum(
            (w_dense._blocks >= BlockType.COAL_ORE)
            & (w_dense._blocks <= BlockType.COPPER_ORE)
        )
        assert dense_count > sparse_count


class TestFillerBlocks:
    """Filler block placement tests."""

    def test_deepslate_present(self) -> None:
        w = World(
            size=(32, 32, 32), seed=42, caves_enabled=False,
        )
        ds_count = int(np.sum(w._blocks == BlockType.DEEPSLATE))
        assert ds_count > 0, "No deepslate found"

    def test_filler_blocks_present(self) -> None:
        w = World(
            size=(32, 32, 32), seed=42,
            ore_density_multiplier=5.0, caves_enabled=False,
        )
        filler_types = [
            BlockType.DIRT, BlockType.GRAVEL,
            BlockType.GRANITE, BlockType.DIORITE,
            BlockType.ANDESITE,
        ]
        total = 0
        for bt in filler_types:
            total += int(np.sum(w._blocks == bt))
        assert total > 0, "No filler blocks found"


class TestBiomeGeneration:
    """Biome map tests."""

    def test_biome_map_shape(self) -> None:
        w = World(
            size=(32, 32, 32), seed=42, caves_enabled=False,
        )
        assert w.biome_map.shape == (32, 32)

    def test_biome_map_has_plains(self) -> None:
        bm = BiomeGenerator.generate_biome_map(64, 64, seed=42)
        assert np.any(bm == BiomeType.PLAINS)

    def test_biome_map_reproducible(self) -> None:
        bm1 = BiomeGenerator.generate_biome_map(32, 32, seed=42)
        bm2 = BiomeGenerator.generate_biome_map(32, 32, seed=42)
        np.testing.assert_array_equal(bm1, bm2)

    def test_biome_map_varies_with_seed(self) -> None:
        bm1 = BiomeGenerator.generate_biome_map(32, 32, seed=1)
        bm2 = BiomeGenerator.generate_biome_map(32, 32, seed=999)
        assert not np.array_equal(bm1, bm2)


class TestCaveGeneration:
    """Cave generation tests."""

    def test_caves_produce_air(self) -> None:
        w = World(size=(64, 64, 64), seed=42, caves_enabled=True)
        air_count = int(np.sum(w._blocks == BlockType.AIR))
        assert air_count > 0

    def test_no_caves_when_disabled(self) -> None:
        w = World(
            size=(32, 32, 32), seed=42, caves_enabled=False,
        )
        air_count = int(np.sum(w._blocks == BlockType.AIR))
        assert air_count == 0

    def test_caves_dont_remove_bedrock(self) -> None:
        w = World(
            size=(32, 32, 32), seed=42, caves_enabled=True,
        )
        bedrock_count = int(
            np.sum(w._blocks[:, 0, :] == BlockType.BEDROCK)
        )
        assert bedrock_count == 32 * 32


class TestWorldAccess:
    """Block access and modification tests."""

    def test_get_set_block(self) -> None:
        w = World(size=(8, 8, 8), seed=1, caves_enabled=False)
        w.set_block(3, 3, 3, BlockType.AIR)
        assert w.get_block(3, 3, 3) == BlockType.AIR

    def test_indexing(self) -> None:
        w = World(size=(8, 8, 8), seed=1, caves_enabled=False)
        w[3, 3, 3] = BlockType.DIAMOND_ORE
        assert w[3, 3, 3] == BlockType.DIAMOND_ORE

    def test_get_local_cube_shape(self) -> None:
        w = World(
            size=(16, 16, 16), seed=1, caves_enabled=False,
        )
        pos = np.array([8, 8, 8])
        cube = w.get_local_cube(pos, radius=2)
        assert cube.shape == (5, 5, 5)

    def test_get_local_cube_boundary(self) -> None:
        w = World(size=(8, 8, 8), seed=1, caves_enabled=False)
        pos = np.array([0, 0, 0])
        cube = w.get_local_cube(pos, radius=2)
        assert cube.shape == (5, 5, 5)
        # Out-of-bounds should be BEDROCK
        assert cube[0, 0, 0] == BlockType.BEDROCK


class TestSlidingWindow:
    """Sliding window observation extraction tests."""

    def test_shape(self) -> None:
        w = World(size=(32, 64, 32), seed=42, caves_enabled=False)
        pos = np.array([16, 32, 16])
        window = w.get_sliding_window(
            pos,
            radius_xz=OBS_WINDOW_RADIUS_XZ,
            y_above=OBS_WINDOW_Y_ABOVE,
            y_below=OBS_WINDOW_Y_BELOW,
        )
        assert window.shape == (OBS_WINDOW_X, OBS_WINDOW_Y, OBS_WINDOW_Z)

    def test_boundary_fill(self) -> None:
        """Out-of-bounds positions should be filled with BEDROCK."""
        w = World(size=(16, 16, 16), seed=42, caves_enabled=False)
        # Place at corner so most of window is out-of-bounds
        pos = np.array([0, 0, 0])
        window = w.get_sliding_window(
            pos,
            radius_xz=OBS_WINDOW_RADIUS_XZ,
            y_above=OBS_WINDOW_Y_ABOVE,
            y_below=OBS_WINDOW_Y_BELOW,
        )
        assert window.shape == (OBS_WINDOW_X, OBS_WINDOW_Y, OBS_WINDOW_Z)
        # Most of the window below should be bedrock fill
        assert np.sum(window == BlockType.BEDROCK) > 0

    def test_center_matches_world(self) -> None:
        """Centre voxel of the window should match the world block."""
        w = World(size=(32, 64, 32), seed=42, caves_enabled=False)
        pos = np.array([16, 32, 16])
        w.set_block(16, 32, 16, BlockType.AIR)
        window = w.get_sliding_window(
            pos,
            radius_xz=OBS_WINDOW_RADIUS_XZ,
            y_above=OBS_WINDOW_Y_ABOVE,
            y_below=OBS_WINDOW_Y_BELOW,
        )
        # Centre in window coords:
        # x-centre = radius_xz, y-centre = y_below, z-centre = radius_xz
        cx = OBS_WINDOW_RADIUS_XZ
        cy = OBS_WINDOW_Y_BELOW
        cz = OBS_WINDOW_RADIUS_XZ
        assert window[cx, cy, cz] == BlockType.AIR

    def test_dtype_int8(self) -> None:
        w = World(size=(16, 32, 16), seed=42, caves_enabled=False)
        pos = np.array([8, 16, 8])
        window = w.get_sliding_window(pos)
        assert window.dtype == np.int8


class TestWorldReset:
    """World reset and seed tests."""

    def test_different_seeds_different_worlds(self) -> None:
        w1 = World(
            size=(16, 16, 16), seed=1, caves_enabled=False,
        )
        w2 = World(
            size=(16, 16, 16), seed=999, caves_enabled=False,
        )
        assert not np.array_equal(w1._blocks, w2._blocks)

    def test_reset_regenerates(self) -> None:
        w = World(
            size=(16, 16, 16), seed=1, caves_enabled=False,
        )
        blocks_before = w._blocks.copy()
        w.reset(seed=999)
        assert not np.array_equal(blocks_before, w._blocks)

    def test_same_seed_reproducible(self) -> None:
        w1 = World(
            size=(16, 16, 16), seed=42, caves_enabled=False,
        )
        w2 = World(
            size=(16, 16, 16), seed=42, caves_enabled=False,
        )
        assert np.array_equal(w1._blocks, w2._blocks)


class TestPerformance:
    """Performance benchmarks."""

    def test_generation_under_2s(self) -> None:
        # Warm up with same array size
        _ = World(size=(64, 64, 64), seed=999, caves_enabled=True)

        times = []
        for i in range(3):
            t0 = time.time()
            _ = World(
                size=(64, 64, 64), seed=i, caves_enabled=True,
            )
            times.append(time.time() - t0)
        best = min(times)
        assert best < 2.0, (
            f"64x64x64 generation took {best:.3f}s (limit: 2.0s)"
        )
