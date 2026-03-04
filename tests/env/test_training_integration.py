"""Tests for make_training_env with real chunks and mixed mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from prospect_rl.config import BiomeType, BlockType


@pytest.fixture
def cache_dir(tmp_path: Path) -> str:
    """Create a cache dir with synthetic .npz files for testing."""
    size = (32, 64, 32)
    rng = np.random.default_rng(42)

    for i in range(3):
        blocks = np.full(size, BlockType.STONE, dtype=np.int8)
        blocks[:, 0, :] = BlockType.BEDROCK

        # Add ores
        ore_mask = rng.random(size) < 0.02
        ore_mask[:, 0, :] = False
        ore_types = [
            BlockType.COAL_ORE, BlockType.IRON_ORE,
            BlockType.DIAMOND_ORE, BlockType.GOLD_ORE,
            BlockType.REDSTONE_ORE, BlockType.EMERALD_ORE,
            BlockType.LAPIS_ORE, BlockType.COPPER_ORE,
        ]
        choices = rng.choice(
            [int(o) for o in ore_types],
            size=int(ore_mask.sum()),
        )
        blocks[ore_mask] = choices.astype(np.int8)

        biome_map = np.full(
            (size[0], size[2]), BiomeType.PLAINS, dtype=np.int8,
        )

        np.savez_compressed(
            tmp_path / f"{i:05d}.npz",
            blocks=blocks,
            biome_map=biome_map,
        )

    return str(tmp_path)


class TestMakeTrainingEnvWithRealChunks:
    """Test make_training_env with world_class and cache_dir params."""

    def test_accepts_world_class_and_cache_dir(
        self, cache_dir: str,
    ) -> None:
        from prospect_rl.env.world.real_chunk_world import RealChunkWorld
        from prospect_rl.models.ppo_config import make_training_env

        env = make_training_env(
            n_envs=2,
            stage_index=5,
            seed=42,
            world_class=RealChunkWorld,
            cache_dir=cache_dir,
        )
        obs = env.reset()
        assert obs["voxels"].shape[0] == 2  # n_envs

        # Step once
        actions = np.array([0, 0])  # FORWARD
        obs, rewards, dones, infos = env.step(actions)
        assert obs["voxels"].shape[0] == 2

    def test_default_no_world_class_works(self) -> None:
        from prospect_rl.models.ppo_config import make_training_env

        env = make_training_env(n_envs=2, stage_index=0, seed=42)
        obs = env.reset()
        assert obs["voxels"].shape[0] == 2


class TestMixedTrainingEnv:
    """Test mixed sim + real chunk environments."""

    def test_mixed_env_creation(self, cache_dir: str) -> None:
        from prospect_rl.models.ppo_config import make_training_env

        env = make_training_env(
            n_envs=4,
            stage_index=0,
            seed=42,
            real_fraction=0.5,
            real_cache_dir=cache_dir,
        )
        obs = env.reset()
        assert obs["voxels"].shape[0] == 4

    def test_mixed_env_steps(self, cache_dir: str) -> None:
        from prospect_rl.models.ppo_config import make_training_env

        env = make_training_env(
            n_envs=4,
            stage_index=0,
            seed=42,
            real_fraction=0.5,
            real_cache_dir=cache_dir,
        )
        env.reset()

        for _ in range(5):
            actions = np.array([0, 0, 0, 0])
            obs, rewards, dones, infos = env.step(actions)

        assert obs["voxels"].shape[0] == 4

    def test_no_real_fraction_all_sim(self) -> None:
        from prospect_rl.models.ppo_config import make_training_env

        env = make_training_env(
            n_envs=4,
            stage_index=0,
            seed=42,
            real_fraction=0.0,
        )
        obs = env.reset()
        assert obs["voxels"].shape[0] == 4

    def test_world_type_varies_in_mixed_env(
        self, cache_dir: str,
    ) -> None:
        from prospect_rl.models.ppo_config import make_training_env

        env = make_training_env(
            n_envs=4,
            stage_index=0,
            seed=42,
            real_fraction=0.5,
            real_cache_dir=cache_dir,
        )
        env.reset()

        # Step and check infos
        actions = np.array([0, 0, 0, 0])
        obs, rewards, dones, infos = env.step(actions)

        world_types = [info.get("world_type", "sim") for info in infos]
        # First 2 should be sim, last 2 should be real
        assert world_types[:2] == ["sim", "sim"]
        assert world_types[2:] == ["real", "real"]
