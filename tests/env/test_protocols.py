"""Tests for WorldBackend and TurtleBackend protocol compliance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from prospect_rl.config import BiomeType, BlockType
from prospect_rl.env.world.protocols import TurtleBackend, WorldBackend


@pytest.fixture
def cache_dir(tmp_path: Path) -> str:
    """Create a cache dir with a single synthetic .npz file."""
    size = (16, 32, 16)
    rng = np.random.default_rng(42)

    blocks = np.full(size, BlockType.STONE, dtype=np.int8)
    blocks[:, 0, :] = BlockType.BEDROCK
    ore_mask = rng.random(size) < 0.02
    ore_mask[:, 0, :] = False
    blocks[ore_mask] = BlockType.COAL_ORE
    biome_map = np.full((size[0], size[2]), BiomeType.PLAINS, dtype=np.int8)

    np.savez_compressed(
        tmp_path / "00000.npz",
        blocks=blocks,
        biome_map=biome_map,
    )
    return str(tmp_path)


class TestWorldBackendProtocol:
    """Verify that all World implementations satisfy WorldBackend."""

    def test_world_satisfies_protocol(self) -> None:
        from prospect_rl.env.world.world import World

        w = World(size=(10, 10, 10), seed=1)
        assert isinstance(w, WorldBackend)

    def test_real_chunk_world_satisfies_protocol(
        self, cache_dir: str,
    ) -> None:
        from prospect_rl.env.world.real_chunk_world import RealChunkWorld

        w = RealChunkWorld(size=(16, 32, 16), seed=1, cache_dir=cache_dir)
        assert isinstance(w, WorldBackend)

    def test_stub_world_is_minimal_fallback(self) -> None:
        """_StubWorld is a minimal fallback — it lacks get_block/set_block."""
        from prospect_rl.env.mining_env import _StubWorld

        rng = np.random.default_rng(42)
        w = _StubWorld(size=(10, 10, 10), rng=rng)
        # _StubWorld intentionally omits get_block/set_block
        # so it does not satisfy the full WorldBackend protocol
        assert not isinstance(w, WorldBackend)


class TestTurtleBackendProtocol:
    """Verify that Turtle satisfies TurtleBackend."""

    def test_turtle_satisfies_protocol(self) -> None:
        from prospect_rl.env.turtle import Turtle

        t = Turtle(position=np.array([5, 5, 5]))
        assert isinstance(t, TurtleBackend)
