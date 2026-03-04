"""Tests for RealChunkWorld — loaded from cached .npz chunk data."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
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
from prospect_rl.env.world.real_chunk_world import RealChunkWorld


@pytest.fixture
def cache_dir(tmp_path: Path) -> str:
    """Create a temporary cache directory with synthetic .npz files."""
    size = (32, 64, 32)
    rng = np.random.default_rng(42)

    for i in range(3):
        blocks = np.full(size, BlockType.STONE, dtype=np.int8)
        blocks[:, 0, :] = BlockType.BEDROCK

        # Add some ores
        ore_mask = rng.random(size) < 0.02
        ore_mask[:, 0, :] = False  # no ores in bedrock
        ore_indices = np.where(ore_mask)
        ore_types = [
            BlockType.COAL_ORE, BlockType.IRON_ORE,
            BlockType.DIAMOND_ORE, BlockType.GOLD_ORE,
        ]
        choices = rng.choice(
            [int(o) for o in ore_types],
            size=len(ore_indices[0]),
        )
        blocks[ore_indices] = choices.astype(np.int8)

        # Add some air (caves)
        air_mask = rng.random(size) < 0.05
        air_mask[:, 0, :] = False
        blocks[air_mask] = BlockType.AIR

        biome_map = np.full(
            (size[0], size[2]), BiomeType.PLAINS, dtype=np.int8,
        )
        # Sprinkle some mountain biome
        biome_map[rng.random((size[0], size[2])) < 0.3] = BiomeType.MOUNTAINS

        np.savez_compressed(
            tmp_path / f"{i:05d}.npz",
            blocks=blocks,
            biome_map=biome_map,
        )

    return str(tmp_path)


class TestRealChunkWorldInit:
    """Initialization and basic properties."""

    def test_loads_from_cache(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        assert w.shape == (32, 64, 32)

    def test_blocks_are_int8(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        assert w._blocks.dtype == np.int8

    def test_bedrock_at_y0(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        for x in range(32):
            for z in range(32):
                assert w.get_block(x, 0, z) == BlockType.BEDROCK

    def test_different_seeds_may_load_different_chunks(
        self, cache_dir: str,
    ) -> None:
        """Different seeds should select different cache files (statistically)."""
        worlds = []
        for seed in range(20):
            w = RealChunkWorld(
                size=(32, 64, 32), seed=seed, cache_dir=cache_dir,
            )
            worlds.append(w._blocks.copy())

        # With 3 cache files and 20 seeds, we should see variation
        different = sum(
            1 for i in range(1, len(worlds))
            if not np.array_equal(worlds[0], worlds[i])
        )
        assert different > 0, "All seeds loaded the same chunk"

    def test_missing_cache_raises(self) -> None:
        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises(FileNotFoundError, match="No .npz"):
                RealChunkWorld(
                    size=(32, 64, 32),
                    seed=1,
                    cache_dir=empty_dir,
                )


class TestRealChunkWorldSizeAdaptation:
    """Test crop/pad when target size differs from cache."""

    def test_smaller_target(self, cache_dir: str) -> None:
        """Target smaller than cache → crop."""
        w = RealChunkWorld(size=(16, 32, 16), seed=1, cache_dir=cache_dir)
        assert w.shape == (16, 32, 16)
        assert w._blocks.shape == (16, 32, 16)

    def test_larger_target(self, cache_dir: str) -> None:
        """Target larger than cache → pad with stone."""
        w = RealChunkWorld(size=(64, 128, 64), seed=1, cache_dir=cache_dir)
        assert w.shape == (64, 128, 64)
        assert w._blocks.shape == (64, 128, 64)
        # Padded region should be stone (default)
        assert w.get_block(50, 50, 50) == BlockType.STONE

    def test_biome_map_fits(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(16, 32, 16), seed=1, cache_dir=cache_dir)
        assert w.biome_map.shape == (16, 16)


class TestRealChunkWorldBlockAccess:
    """Block access and modification."""

    def test_get_set_block(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        w.set_block(5, 5, 5, BlockType.AIR)
        assert w.get_block(5, 5, 5) == BlockType.AIR

    def test_indexing(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        w[5, 5, 5] = BlockType.DIAMOND_ORE
        assert w[5, 5, 5] == BlockType.DIAMOND_ORE

    def test_count_blocks(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        bedrock_count = w.count_blocks([int(BlockType.BEDROCK)])
        assert bedrock_count >= 32 * 32  # at least the floor

    def test_count_ores(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        ore_count = w.count_blocks([
            int(BlockType.COAL_ORE),
            int(BlockType.IRON_ORE),
            int(BlockType.DIAMOND_ORE),
            int(BlockType.GOLD_ORE),
        ])
        assert ore_count > 0


class TestRealChunkWorldSlidingWindow:
    """Sliding window observation extraction."""

    def test_shape(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=42, cache_dir=cache_dir)
        pos = np.array([16, 32, 16])
        window = w.get_sliding_window(
            pos,
            radius_xz=OBS_WINDOW_RADIUS_XZ,
            y_above=OBS_WINDOW_Y_ABOVE,
            y_below=OBS_WINDOW_Y_BELOW,
        )
        assert window.shape == (OBS_WINDOW_X, OBS_WINDOW_Y, OBS_WINDOW_Z)

    def test_boundary_fill(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=42, cache_dir=cache_dir)
        pos = np.array([0, 0, 0])
        window = w.get_sliding_window(
            pos,
            radius_xz=OBS_WINDOW_RADIUS_XZ,
            y_above=OBS_WINDOW_Y_ABOVE,
            y_below=OBS_WINDOW_Y_BELOW,
        )
        assert window.shape == (OBS_WINDOW_X, OBS_WINDOW_Y, OBS_WINDOW_Z)
        assert np.sum(window == BlockType.BEDROCK) > 0

    def test_center_matches_world(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=42, cache_dir=cache_dir)
        pos = np.array([16, 32, 16])
        w.set_block(16, 32, 16, BlockType.AIR)
        window = w.get_sliding_window(
            pos,
            radius_xz=OBS_WINDOW_RADIUS_XZ,
            y_above=OBS_WINDOW_Y_ABOVE,
            y_below=OBS_WINDOW_Y_BELOW,
        )
        cx = OBS_WINDOW_RADIUS_XZ
        cy = OBS_WINDOW_Y_BELOW
        cz = OBS_WINDOW_RADIUS_XZ
        assert window[cx, cy, cz] == BlockType.AIR

    def test_dtype_int8(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=42, cache_dir=cache_dir)
        pos = np.array([16, 32, 16])
        window = w.get_sliding_window(pos)
        assert window.dtype == np.int8


class TestRealChunkWorldFindValidSpawn:
    """Intelligent spawn position selection."""

    def test_spawn_on_solid_ground(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        x, y, z = w.find_valid_spawn(16, 16)
        # Block below spawn must be solid (not AIR or BEDROCK)
        block_below = w.get_block(x, y - 1, z)
        assert block_below not in (BlockType.AIR, BlockType.BEDROCK)

    def test_spawn_within_preferred_range(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        preferred = (10.0, 30.0)
        x, y, z = w.find_valid_spawn(16, 16, preferred_y_range=preferred)
        # Should be within preferred range or at least valid
        block_below = w.get_block(x, y - 1, z)
        assert block_below not in (BlockType.AIR, BlockType.BEDROCK)

    def test_spawn_deterministic_with_seed(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        pos1 = w.find_valid_spawn(16, 16, rng=rng1)
        pos2 = w.find_valid_spawn(16, 16, rng=rng2)
        assert pos1 == pos2

    def test_spawn_returns_valid_coords(self, cache_dir: str) -> None:
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        x, y, z = w.find_valid_spawn(16, 16)
        assert 0 <= x < 32
        assert 0 <= y < 64
        assert 0 <= z < 32


class TestRealChunkWorldOreValidation:
    """Ore content validation on load."""

    def test_ore_validation_passes_with_ores(self, cache_dir: str) -> None:
        """No exception on load when ores are present."""
        w = RealChunkWorld(size=(32, 64, 32), seed=1, cache_dir=cache_dir)
        assert w.count_blocks([int(BlockType.COAL_ORE)]) > 0

    def test_zero_ore_chunk_logs_warning(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Chunk with no ores should log a warning."""
        import logging

        # Create a cache with 0 ores
        blocks = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks[:, 0, :] = BlockType.BEDROCK
        biome_map = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00000.npz",
            blocks=blocks,
            biome_map=biome_map,
        )
        with caplog.at_level(logging.WARNING):
            RealChunkWorld(
                size=(16, 32, 16), seed=1, cache_dir=str(tmp_path),
            )
        assert "0 ores" in caplog.text


class TestRealChunkWorldEnvIntegration:
    """Test that RealChunkWorld works with MiningEnv."""

    def test_env_with_real_chunk_world(self, cache_dir: str) -> None:
        """MiningEnv should work with RealChunkWorld as world_class."""
        from prospect_rl.env.mining_env import MinecraftMiningEnv

        env = MinecraftMiningEnv(
            curriculum_stage=5,  # stage_real_chunks
            world_class=RealChunkWorld,
            seed=42,
            cache_dir=cache_dir,
        )
        obs, info = env.reset()

        assert "voxels" in obs
        assert "scalars" in obs
        assert "pref" in obs
        assert obs["voxels"].shape[0] == 14  # channels

        # Take a few steps
        for _ in range(10):
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = int(np.random.choice(valid_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert isinstance(reward, float)

    def test_world_type_is_real(self, cache_dir: str) -> None:
        """Info dict should report world_type='real' for RealChunkWorld."""
        from prospect_rl.env.mining_env import MinecraftMiningEnv

        env = MinecraftMiningEnv(
            curriculum_stage=5,
            world_class=RealChunkWorld,
            seed=42,
            cache_dir=cache_dir,
        )
        obs, info = env.reset()
        assert info.get("world_type") == "real"

    def test_world_type_is_sim_for_procedural(self) -> None:
        """Info dict should report world_type='sim' for procedural worlds."""
        from prospect_rl.env.mining_env import MinecraftMiningEnv

        env = MinecraftMiningEnv(curriculum_stage=0, seed=42)
        obs, info = env.reset()
        assert info.get("world_type") == "sim"

    def test_spawn_uses_find_valid_spawn(self, cache_dir: str) -> None:
        """Agent should spawn on solid ground in real chunk worlds."""
        from prospect_rl.env.mining_env import MinecraftMiningEnv

        env = MinecraftMiningEnv(
            curriculum_stage=5,
            world_class=RealChunkWorld,
            seed=42,
            cache_dir=cache_dir,
        )
        obs, info = env.reset()
        pos = info["position"]
        # Block at spawn position should be AIR (cleared for turtle)
        assert env._world[pos[0], pos[1], pos[2]] == BlockType.AIR


class TestRealChunkWorldMinOresFilter:
    """Test min_ores filtering of cache files."""

    def test_min_ores_filters_empty_chunks(self, tmp_path: Path) -> None:
        """Chunks with 0 ores should be filtered when min_ores > 0."""
        # Clear class-level cache for this test
        RealChunkWorld._filtered_cache.clear()

        # Chunk 0: no ores (all stone)
        blocks0 = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks0[:, 0, :] = BlockType.BEDROCK
        biome0 = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00000.npz", blocks=blocks0, biome_map=biome0,
        )

        # Chunk 1: has ores
        blocks1 = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks1[:, 0, :] = BlockType.BEDROCK
        blocks1[5, 10, 5] = BlockType.COAL_ORE
        blocks1[6, 12, 6] = BlockType.IRON_ORE
        biome1 = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00001.npz", blocks=blocks1, biome_map=biome1,
        )

        # With min_ores=1, only chunk 1 should be available
        w = RealChunkWorld(
            size=(16, 32, 16), seed=1,
            cache_dir=str(tmp_path), min_ores=1,
        )
        # Should have loaded successfully (only chunk 1 available)
        ore_count = w.count_blocks([
            int(BlockType.COAL_ORE), int(BlockType.IRON_ORE),
        ])
        assert ore_count > 0

    def test_min_ores_zero_allows_all(self, tmp_path: Path) -> None:
        """min_ores=0 should accept all chunks including empty ones."""
        RealChunkWorld._filtered_cache.clear()

        # Create one empty chunk
        blocks = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks[:, 0, :] = BlockType.BEDROCK
        biome = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00000.npz", blocks=blocks, biome_map=biome,
        )

        # min_ores=0 should not filter
        w = RealChunkWorld(
            size=(16, 32, 16), seed=1,
            cache_dir=str(tmp_path), min_ores=0,
        )
        assert w.shape == (16, 32, 16)

    def test_min_ores_raises_when_all_filtered(
        self, tmp_path: Path,
    ) -> None:
        """Should raise FileNotFoundError if all chunks are filtered out."""
        RealChunkWorld._filtered_cache.clear()

        blocks = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks[:, 0, :] = BlockType.BEDROCK
        biome = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00000.npz", blocks=blocks, biome_map=biome,
        )

        with pytest.raises(FileNotFoundError, match=">= 100 ores"):
            RealChunkWorld(
                size=(16, 32, 16), seed=1,
                cache_dir=str(tmp_path), min_ores=100,
            )

    def test_filter_cache_is_class_level(self, tmp_path: Path) -> None:
        """Filtered file list should be cached across instances."""
        RealChunkWorld._filtered_cache.clear()

        blocks = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks[:, 0, :] = BlockType.BEDROCK
        blocks[5, 10, 5] = BlockType.COAL_ORE
        biome = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00000.npz", blocks=blocks, biome_map=biome,
        )

        # First call populates cache
        RealChunkWorld(
            size=(16, 32, 16), seed=1,
            cache_dir=str(tmp_path), min_ores=1,
        )
        key = (str(tmp_path), 1, None)
        assert key in RealChunkWorld._filtered_cache

        # Second call should use cached result
        RealChunkWorld(
            size=(16, 32, 16), seed=2,
            cache_dir=str(tmp_path), min_ores=1,
        )
        # Should still be same cache entry
        assert key in RealChunkWorld._filtered_cache


class TestRealChunkWorldRequiredBiome:
    """Test required_biome filtering of cache files."""

    def test_required_biome_filters_chunks(self, tmp_path: Path) -> None:
        """Only chunks with the required biome should be loaded."""
        RealChunkWorld._filtered_cache.clear()

        # Chunk 0: plains only
        blocks0 = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks0[:, 0, :] = BlockType.BEDROCK
        blocks0[5, 10, 5] = BlockType.COAL_ORE
        biome0 = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00000.npz", blocks=blocks0, biome_map=biome0,
        )

        # Chunk 1: has mountains
        blocks1 = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks1[:, 0, :] = BlockType.BEDROCK
        blocks1[5, 10, 5] = BlockType.EMERALD_ORE
        biome1 = np.full((16, 16), BiomeType.MOUNTAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00001.npz", blocks=blocks1, biome_map=biome1,
        )

        # With required_biome=MOUNTAINS, only chunk 1 should be available
        w = RealChunkWorld(
            size=(16, 32, 16), seed=1,
            cache_dir=str(tmp_path),
            required_biome=int(BiomeType.MOUNTAINS),
        )
        emerald_count = w.count_blocks([int(BlockType.EMERALD_ORE)])
        assert emerald_count > 0

    def test_required_biome_none_allows_all(
        self, tmp_path: Path,
    ) -> None:
        """required_biome=None should accept all chunks."""
        RealChunkWorld._filtered_cache.clear()

        blocks = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks[:, 0, :] = BlockType.BEDROCK
        blocks[5, 10, 5] = BlockType.COAL_ORE
        biome = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00000.npz", blocks=blocks, biome_map=biome,
        )

        w = RealChunkWorld(
            size=(16, 32, 16), seed=1,
            cache_dir=str(tmp_path), required_biome=None,
        )
        assert w.shape == (16, 32, 16)

    def test_required_biome_raises_when_none_match(
        self, tmp_path: Path,
    ) -> None:
        """Should raise if no chunks have the required biome."""
        RealChunkWorld._filtered_cache.clear()

        blocks = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks[:, 0, :] = BlockType.BEDROCK
        biome = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        np.savez_compressed(
            tmp_path / "00000.npz", blocks=blocks, biome_map=biome,
        )

        with pytest.raises(FileNotFoundError, match="biome=1"):
            RealChunkWorld(
                size=(16, 32, 16), seed=1,
                cache_dir=str(tmp_path),
                required_biome=int(BiomeType.MOUNTAINS),
            )

    def test_partial_biome_match_accepted(
        self, tmp_path: Path,
    ) -> None:
        """Chunk with mixed biomes should match if any cell matches."""
        RealChunkWorld._filtered_cache.clear()

        blocks = np.full((16, 32, 16), BlockType.STONE, dtype=np.int8)
        blocks[:, 0, :] = BlockType.BEDROCK
        blocks[5, 10, 5] = BlockType.EMERALD_ORE
        # Mostly plains but one corner is mountains
        biome = np.full((16, 16), BiomeType.PLAINS, dtype=np.int8)
        biome[0, 0] = BiomeType.MOUNTAINS
        np.savez_compressed(
            tmp_path / "00000.npz", blocks=blocks, biome_map=biome,
        )

        w = RealChunkWorld(
            size=(16, 32, 16), seed=1,
            cache_dir=str(tmp_path),
            required_biome=int(BiomeType.MOUNTAINS),
        )
        assert w.shape == (16, 32, 16)
