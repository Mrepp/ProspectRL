"""World backed by pre-cached real Minecraft chunk data.

``RealChunkWorld`` is a drop-in replacement for ``World``.  It loads a
random cached ``.npz`` file (produced by ``prospect_rl.tools.cache_chunks``)
instead of running the procedural generation pipeline.

At training time no Amulet dependency is needed — the cache files are
pure numpy arrays.
"""

from __future__ import annotations

import glob
import logging
from typing import Any

import numpy as np
from prospect_rl.config import BlockType, ORE_TYPES

logger = logging.getLogger(__name__)


class RealChunkWorld:
    """3D block grid loaded from cached real Minecraft chunks.

    The public API mirrors ``World`` exactly so that ``MiningEnv`` can
    use either class interchangeably.

    Parameters
    ----------
    size:
        Target ``(sx, sy, sz)`` grid dimensions.  If the cached data
        doesn't match, it is cropped or padded.
    seed:
        Seed for the RNG that picks which cached chunk to load.
    cache_dir:
        Directory containing ``.npz`` files produced by
        ``prospect_rl.tools.cache_chunks``.
    min_ores:
        Skip cached chunks with fewer than this many ore blocks.
        Set to 0 to accept all chunks (default).
    required_biome:
        Only load chunks whose biome map contains at least one cell
        of this ``BiomeType`` value.  ``None`` means no biome filter.
    """

    # Class-level cache: (cache_dir, min_ores, required_biome) -> file list
    _filtered_cache: dict[tuple[str, int, int | None], list[str]] = {}

    def __init__(
        self,
        size: tuple[int, int, int] = (64, 128, 64),
        seed: int = 42,
        cache_dir: str = "data/chunk_cache/default",
        min_ores: int = 0,
        required_biome: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._size = size
        self._rng = np.random.default_rng(seed)

        # Discover and filter cache files
        cache_files = self._get_filtered_files(
            cache_dir, min_ores, required_biome,
        )
        if not cache_files:
            biome_msg = (
                f" with biome={required_biome}" if required_biome is not None
                else ""
            )
            raise FileNotFoundError(
                f"No .npz cache files found in {cache_dir!r} with "
                f">= {min_ores} ores{biome_msg}.  "
                "Run `python -m prospect_rl.tools.cache_chunks` first."
            )

        # Pick a random cached chunk
        chosen = self._rng.choice(cache_files)
        data = np.load(chosen, allow_pickle=False)

        raw_blocks = data["blocks"]      # (cx, cy, cz) int8
        raw_biomes = data["biome_map"]   # (cx, cz) int8

        # Fit to target size
        self._blocks = self._fit_blocks(raw_blocks, size)
        self._biome_map = self._fit_biome_map(
            raw_biomes, (size[0], size[2]),
        )

        # Enforce bedrock floor at y=0
        self._blocks[:, 0, :] = BlockType.BEDROCK

        # Validate ore content
        ore_ids = [int(o) for o in ORE_TYPES]
        total_ores = int(np.isin(self._blocks, ore_ids).sum())
        if total_ores == 0:
            logger.warning(
                "Loaded chunk from %s has 0 ores — "
                "preference fallback will be needed.", chosen,
            )

        logger.debug(
            "Loaded real chunk from %s (raw %s → %s, %d ores)",
            chosen, raw_blocks.shape, self._blocks.shape, total_ores,
        )

    # ------------------------------------------------------------------
    # Public API (mirrors World exactly)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._size

    @property
    def biome_map(self) -> np.ndarray:
        """2D biome map of shape ``(sx, sz)``."""
        return self._biome_map

    def reset(self, seed: int | None = None) -> None:
        """Re-load a different cached chunk.

        Note: ``MiningEnv`` creates a fresh world object on each
        ``reset()`` call, so this method is rarely used directly.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # The constructor loads from cache; for reset we'd need the
        # cache_dir which we don't store.  In practice MiningEnv
        # instantiates a new RealChunkWorld each episode.

    def get_block(self, x: int, y: int, z: int) -> int:
        """Return the block type at ``(x, y, z)``."""
        return int(self._blocks[x, y, z])

    def set_block(self, x: int, y: int, z: int, block_id: int) -> None:
        """Set the block at ``(x, y, z)``."""
        self._blocks[x, y, z] = np.int8(block_id)

    def count_blocks(self, block_ids: list[int]) -> int:
        """Count total blocks matching any of the given IDs."""
        mask = np.isin(
            self._blocks, np.array(block_ids, dtype=np.int8),
        )
        return int(np.sum(mask))

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
        Identical implementation to ``World.get_sliding_window``.
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

    def get_local_cube(
        self, pos: np.ndarray, radius: int = 2,
    ) -> np.ndarray:
        """Extract a ``(2r+1)^3`` cube centred on *pos*."""
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

    # numpy-style indexing
    def __getitem__(self, key: Any) -> Any:
        return self._blocks[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._blocks[key] = value

    # ------------------------------------------------------------------
    # Spawn helpers
    # ------------------------------------------------------------------

    def find_valid_spawn(
        self,
        center_x: int,
        center_z: int,
        preferred_y_range: tuple[float, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int, int]:
        """Find a spawn position on solid ground.

        Scans the column at ``(center_x ± jitter, center_z ± jitter)``
        for a block with a solid block below it, preferring positions
        within *preferred_y_range*.

        Parameters
        ----------
        center_x, center_z:
            Approximate XZ spawn location.
        preferred_y_range:
            Optional ``(y_lo, y_hi)`` in sim coords.  The method
            tries to find a valid spawn within this range first.
        rng:
            Random generator for XZ jitter.

        Returns
        -------
        tuple[int, int, int]
            ``(x, y, z)`` spawn position.
        """
        rng = rng or np.random.default_rng()
        sx, sy, sz = self._size

        # Jitter XZ slightly
        x = int(np.clip(center_x + rng.integers(-3, 4), 1, sx - 2))
        z = int(np.clip(center_z + rng.integers(-3, 4), 1, sz - 2))

        # Define Y search range
        if preferred_y_range is not None:
            y_lo = max(2, int(preferred_y_range[0]))
            y_hi = min(sy - 2, int(preferred_y_range[1]))
        else:
            y_lo = 2
            y_hi = sy - 2

        # Scan downward within preferred range for solid ground
        for y in range(y_hi, y_lo - 1, -1):
            block_below = int(self._blocks[x, y - 1, z])
            if block_below not in (BlockType.AIR, BlockType.BEDROCK):
                return (x, y, z)

        # Expand search to full world height
        for y in range(sy - 2, 1, -1):
            block_below = int(self._blocks[x, y - 1, z])
            if block_below not in (BlockType.AIR, BlockType.BEDROCK):
                return (x, y, z)

        # Last resort: center of world
        return (x, sy // 2, z)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_filtered_files(
        cls,
        cache_dir: str,
        min_ores: int,
        required_biome: int | None = None,
    ) -> list[str]:
        """Return cache file paths, filtered by ore count and biome.

        Results are cached at class level so repeated calls (e.g.
        per-episode resets) don't re-scan the cache directory.
        """
        key = (cache_dir, min_ores, required_biome)
        if key in cls._filtered_cache:
            return cls._filtered_cache[key]

        all_files = sorted(glob.glob(f"{cache_dir}/*.npz"))
        if min_ores <= 0 and required_biome is None:
            cls._filtered_cache[key] = all_files
            return all_files

        ore_ids = np.array([int(o) for o in ORE_TYPES], dtype=np.int8)
        filtered: list[str] = []
        for f in all_files:
            data = np.load(f, allow_pickle=False)
            # Ore count filter
            if min_ores > 0:
                ore_count = int(
                    np.isin(data["blocks"], ore_ids).sum(),
                )
                if ore_count < min_ores:
                    continue
            # Biome filter
            if required_biome is not None:
                biome_map = data["biome_map"]
                if not np.any(biome_map == required_biome):
                    continue
            filtered.append(f)

        logger.info(
            "Filtered %d/%d chunks (min_ores=%d, biome=%s) in %s",
            len(filtered), len(all_files), min_ores,
            required_biome, cache_dir,
        )
        cls._filtered_cache[key] = filtered
        return filtered

    @staticmethod
    def _fit_blocks(
        raw: np.ndarray,
        target_size: tuple[int, int, int],
    ) -> np.ndarray:
        """Crop or pad *raw* blocks to *target_size*."""
        tx, ty, tz = target_size
        rx, ry, rz = raw.shape

        # Start with stone-filled target
        blocks = np.full(target_size, BlockType.STONE, dtype=np.int8)

        # Copy the overlapping region
        cx = min(tx, rx)
        cy = min(ty, ry)
        cz = min(tz, rz)
        blocks[:cx, :cy, :cz] = raw[:cx, :cy, :cz]

        return blocks

    @staticmethod
    def _fit_biome_map(
        raw: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Crop or pad *raw* biome map to *target_size*."""
        from prospect_rl.env.world.mc_block_mapping import DEFAULT_BIOME

        tx, tz = target_size
        rx, rz = raw.shape

        biomes = np.full(target_size, DEFAULT_BIOME, dtype=np.int8)
        cx = min(tx, rx)
        cz = min(tz, rz)
        biomes[:cx, :cz] = raw[:cx, :cz]

        return biomes
