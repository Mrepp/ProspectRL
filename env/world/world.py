"""3D voxel world with ore distribution, biomes, and cave generation.

The ``World`` class stores blocks as a numpy ``int8`` array and provides
a generation pipeline: fill stone -> bedrock floor -> biome map ->
deepslate layer -> filler blocks -> place ores -> carve caves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from prospect_rl.config import BlockType, CaveConfig
from prospect_rl.env.world.biome import BiomeGenerator
from prospect_rl.env.world.cave_generation import CaveGenerator
from prospect_rl.env.world.ore_distribution import OreDistributor


class World:
    """3D block grid with configurable world generation.

    Parameters
    ----------
    size:
        ``(sx, sy, sz)`` world dimensions.
    seed:
        Random seed for world generation.
    ore_density_multiplier:
        Scales ore spawn probabilities (>1 = denser).
    caves_enabled:
        Whether to generate caves.
    """

    def __init__(
        self,
        size: tuple[int, int, int] = (64, 64, 64),
        seed: int = 42,
        ore_density_multiplier: float = 1.0,
        caves_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        self._size = size
        self._seed = seed
        self._ore_density_multiplier = ore_density_multiplier
        self._caves_enabled = caves_enabled
        self._cave_config = CaveConfig()

        self._blocks: np.ndarray = np.empty(0, dtype=np.int8)
        self._biome_map: np.ndarray = np.empty(0, dtype=np.int8)
        self._generate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._size

    @property
    def biome_map(self) -> np.ndarray:
        """2D biome map of shape ``(sx, sz)``."""
        return self._biome_map

    def reset(self, seed: int | None = None) -> None:
        """Regenerate the world, optionally with a new seed."""
        if seed is not None:
            self._seed = seed
        self._generate()

    def get_block(self, x: int, y: int, z: int) -> int:
        """Return the block type at ``(x, y, z)``."""
        return int(self._blocks[x, y, z])

    def set_block(
        self, x: int, y: int, z: int, block_id: int,
    ) -> None:
        """Set the block at ``(x, y, z)``."""
        self._blocks[x, y, z] = np.int8(block_id)

    def get_local_cube(
        self, pos: np.ndarray, radius: int = 2,
    ) -> np.ndarray:
        """Extract a ``(2r+1)^3`` cube centred on *pos*.

        Out-of-bounds positions are filled with ``BlockType.BEDROCK``.
        """
        side = 2 * radius + 1
        cube = np.full(
            (side, side, side), BlockType.BEDROCK, dtype=np.int8,
        )

        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        sx, sy, sz = self._size

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    wx, wy, wz = px + dx, py + dy, pz + dz
                    if (
                        0 <= wx < sx
                        and 0 <= wy < sy
                        and 0 <= wz < sz
                    ):
                        cube[
                            dx + radius,
                            dy + radius,
                            dz + radius,
                        ] = self._blocks[wx, wy, wz]

        return cube

    def get_sliding_window(
        self,
        pos: np.ndarray,
        radius_xz: int = 4,
        y_above: int = 8,
        y_below: int = 23,
        fill_value: int = BlockType.BEDROCK,
    ) -> np.ndarray:
        """Extract a 3D sliding window centred on *pos*.

        Returns shape ``(window_x, window_y, window_z)`` as int8.
        Out-of-bounds voxels are filled with *fill_value*.
        Fully numpy-vectorized (slice + pad, no Python loops).
        """
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        sx, sy, sz = self._size

        wx = 2 * radius_xz + 1
        wy = y_above + 1 + y_below
        wz = 2 * radius_xz + 1

        x0, x1 = px - radius_xz, px + radius_xz + 1
        y0, y1 = py - y_below, py + y_above + 1
        z0, z1 = pz - radius_xz, pz + radius_xz + 1

        # Padding for out-of-bounds
        px0 = max(0, -x0)
        px1 = max(0, x1 - sx)
        py0 = max(0, -y0)
        py1 = max(0, y1 - sy)
        pz0 = max(0, -z0)
        pz1 = max(0, z1 - sz)

        # Clamp source slice to world bounds
        chunk = self._blocks[
            max(0, x0):min(sx, x1),
            max(0, y0):min(sy, y1),
            max(0, z0):min(sz, z1),
        ]

        if px0 or px1 or py0 or py1 or pz0 or pz1:
            window = np.pad(
                chunk,
                ((px0, px1), (py0, py1), (pz0, pz1)),
                mode="constant",
                constant_values=fill_value,
            )
        else:
            window = chunk.copy()

        assert window.shape == (wx, wy, wz)
        return window

    # numpy-style indexing
    def __getitem__(self, key: Any) -> Any:
        return self._blocks[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._blocks[key] = value

    # ------------------------------------------------------------------
    # Generation pipeline
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """Run the full world generation pipeline."""
        sx, sy, sz = self._size

        # 1. Fill with stone
        self._blocks = np.full(
            (sx, sy, sz), BlockType.STONE, dtype=np.int8,
        )

        # 2. Bedrock floor at y=0
        self._blocks[:, 0, :] = BlockType.BEDROCK

        # 3. Generate biome map
        self._biome_map = BiomeGenerator.generate_biome_map(
            sx, sz, self._seed,
        )

        # 4. Place deepslate below MC y=0
        OreDistributor.place_deepslate(self._blocks, self._size)

        # 5. Place filler blocks (dirt, gravel, granite, etc.)
        OreDistributor.place_filler_blocks(
            self._blocks,
            self._size,
            self._seed,
            density_multiplier=self._ore_density_multiplier,
            biome_map=self._biome_map,
        )

        # 6. Place ores
        OreDistributor.place_ores(
            self._blocks,
            self._size,
            self._seed,
            density_multiplier=self._ore_density_multiplier,
            biome_map=self._biome_map,
        )

        # 7. Carve caves
        if self._caves_enabled:
            CaveGenerator.carve_caves(
                self._blocks,
                self._size,
                self._seed,
                self._cave_config,
            )

        # 8. Re-enforce bedrock floor
        self._blocks[:, 0, :] = BlockType.BEDROCK
