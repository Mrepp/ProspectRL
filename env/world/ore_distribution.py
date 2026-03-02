"""Vectorized ore and filler block placement with Minecraft-parity parameters.

Supports multiple spawn configurations per block type, biome-restricted
spawning, both triangle and uniform vertical distributions, and air-exposure
skipping.  All operations are numpy-vectorized — no Python loops over voxel
coordinates.

Spatial clustering uses a fast cell-based hash noise instead of OpenSimplex
to keep world generation under 2 s for 64^3 worlds.
"""

from __future__ import annotations

import numpy as np
from prospect_rl.config import (
    FILLER_SPAWN_CONFIGS,
    MC_Y_MIN,
    MC_Y_RANGE,
    ORE_SPAWN_CONFIGS,
    BlockType,
    OreSpawnConfig,
)


def _mc_y_to_sim_y(mc_y: int, world_height: int) -> float:
    """Convert a Minecraft y-coordinate to simulation y (proportional)."""
    return (mc_y - MC_Y_MIN) / MC_Y_RANGE * world_height


def _compute_base_probability(cfg: OreSpawnConfig, world_height: int) -> float:
    """Derive per-voxel probability from Minecraft spawn_size and spawn_tries.

    A Minecraft chunk is 16x16 columns.  ``spawn_tries`` attempts are made per
    chunk, each placing ``spawn_size`` blocks across the valid height range.
    We convert this to a continuous per-voxel probability.
    """
    y_min_sim = max(0.0, _mc_y_to_sim_y(cfg.y_min_mc, world_height))
    y_max_sim = min(
        float(world_height),
        _mc_y_to_sim_y(cfg.y_max_mc, world_height),
    )
    y_range = max(y_max_sim - y_min_sim, 1.0)
    return (cfg.spawn_tries * cfg.spawn_size) / (16.0 * 16.0 * y_range)


class OreDistributor:
    """Places ores and filler blocks into a 3D world grid."""

    @staticmethod
    def place_filler_blocks(
        world_blocks: np.ndarray,
        size: tuple[int, int, int],
        seed: int,
        density_multiplier: float = 1.0,
        biome_map: np.ndarray | None = None,
    ) -> np.ndarray:
        """Place filler blocks (dirt, gravel, granite, etc.) in-place.

        Called BEFORE ore placement so ores can overwrite fillers.
        """
        for i, cfg in enumerate(FILLER_SPAWN_CONFIGS):
            _place_single_block(
                world_blocks, cfg, size,
                seed=seed + 5000 + i * 1000,
                density_multiplier=density_multiplier,
                biome_map=biome_map,
                replace_mask=world_blocks == BlockType.STONE,
            )
        return world_blocks

    @staticmethod
    def place_deepslate(
        world_blocks: np.ndarray,
        size: tuple[int, int, int],
    ) -> np.ndarray:
        """Replace stone below MC y=0 with deepslate."""
        sy = size[1]
        deepslate_y = int(_mc_y_to_sim_y(0, sy))
        # clamp: skip bedrock at y=0
        deepslate_y = max(1, min(deepslate_y, sy))
        stone_mask = world_blocks[:, 1:deepslate_y, :] == BlockType.STONE
        world_blocks[:, 1:deepslate_y, :][stone_mask] = np.int8(BlockType.DEEPSLATE)
        return world_blocks

    @staticmethod
    def place_ores(
        world_blocks: np.ndarray,
        size: tuple[int, int, int],
        seed: int,
        density_multiplier: float = 1.0,
        biome_map: np.ndarray | None = None,
        ore_density_overrides: dict[int, float] | None = None,
    ) -> np.ndarray:
        """Place all ore types into the world grid in-place.

        Parameters
        ----------
        world_blocks:
            3D array of shape ``(sx, sy, sz)`` with ``int8`` block IDs.
            Modified in-place.
        size:
            ``(sx, sy, sz)`` world dimensions.
        seed:
            Base seed for noise generation (offset per config).
        density_multiplier:
            Scales the base probability of every ore type.
        biome_map:
            Optional 2D array of shape ``(sx, sz)`` with BiomeType values.
        ore_density_overrides:
            Per-ore multiplier overrides keyed by ``int(BlockType)``.
            When present, completely replaces ``density_multiplier``
            for that ore type.

        Returns
        -------
        The modified ``world_blocks`` array (same reference).
        """
        # Ores can replace stone or deepslate (deepslate variant = same ore)
        replaceable = (
            (world_blocks == BlockType.STONE)
            | (world_blocks == BlockType.DEEPSLATE)
        )

        for i, cfg in enumerate(ORE_SPAWN_CONFIGS):
            effective_mult = density_multiplier
            if ore_density_overrides is not None:
                block_id = int(cfg.block_type)
                if block_id in ore_density_overrides:
                    effective_mult = ore_density_overrides[block_id]

            _place_single_block(
                world_blocks, cfg, size,
                seed=seed + i * 1000,
                density_multiplier=effective_mult,
                biome_map=biome_map,
                replace_mask=replaceable,
            )

        return world_blocks


def _cell_hash_cluster(
    sx: int,
    sy: int,
    sz: int,
    noise_scale: float,
    threshold: float,
    seed: int,
) -> np.ndarray:
    """Fast spatial clustering using cell-based hash noise.

    Divides the world into cells of ``noise_scale`` size, assigns each
    cell a deterministic random value, and thresholds it.  Produces
    vein-like clusters similar to 3D simplex noise but ~1000x faster.
    """
    cell_size = max(1, int(noise_scale))
    # Number of cells along each axis (ceiling division)
    cx = (sx + cell_size - 1) // cell_size
    cy = (sy + cell_size - 1) // cell_size
    cz = (sz + cell_size - 1) // cell_size

    rng = np.random.default_rng(seed + 77777)
    cell_values = rng.random((cx, cy, cz))

    # Map each voxel to its cell
    xi = np.arange(sx) // cell_size
    yi = np.arange(sy) // cell_size
    zi = np.arange(sz) // cell_size

    # Broadcast cell values to full grid
    noise_field = cell_values[
        xi[:, np.newaxis, np.newaxis],
        yi[np.newaxis, :, np.newaxis],
        zi[np.newaxis, np.newaxis, :],
    ]
    return noise_field > threshold


def _place_single_block(
    world_blocks: np.ndarray,
    cfg: OreSpawnConfig,
    size: tuple[int, int, int],
    seed: int,
    density_multiplier: float,
    biome_map: np.ndarray | None,
    replace_mask: np.ndarray,
) -> None:
    """Place a single block config using vectorized operations."""
    sx, sy, sz = size

    # -- 1. Compute simulation Y bounds --
    y_min_sim = max(0, int(_mc_y_to_sim_y(cfg.y_min_mc, sy)))
    y_max_sim = min(sy - 1, int(_mc_y_to_sim_y(cfg.y_max_mc, sy)))

    if y_max_sim <= y_min_sim:
        return  # range doesn't fit in this world

    # -- 2. Vertical probability mask --
    base_prob = _compute_base_probability(cfg, sy) * density_multiplier
    y_all = np.arange(sy, dtype=np.float64)
    vert_prob = np.zeros(sy, dtype=np.float64)

    if cfg.distribution == "uniform":
        mask = (y_all >= y_min_sim) & (y_all <= y_max_sim)
        vert_prob[mask] = base_prob
    else:  # triangle
        if cfg.peak_mc is not None:
            peak = _mc_y_to_sim_y(cfg.peak_mc, sy)
            peak = max(float(y_min_sim), min(float(y_max_sim), peak))
        else:
            peak = (y_min_sim + y_max_sim) / 2.0
        below = (y_all >= y_min_sim) & (y_all <= peak)
        above = (y_all > peak) & (y_all <= y_max_sim)

        span_below = peak - y_min_sim
        span_above = y_max_sim - peak
        if span_below > 0:
            vert_prob[below] = (
                (y_all[below] - y_min_sim) / span_below * base_prob
            )
        if span_above > 0:
            vert_prob[above] = (
                (y_max_sim - y_all[above]) / span_above * base_prob
            )

    # Broadcast to 3D: vert_prob[y] -> (sx, sy, sz)
    vert_mask_3d = np.broadcast_to(
        vert_prob[np.newaxis, :, np.newaxis], (sx, sy, sz)
    )

    # -- 3. Random threshold per voxel --
    rng = np.random.default_rng(seed)
    rand_field = rng.random((sx, sy, sz))
    prob_mask = rand_field < vert_mask_3d

    # -- 4. Spatial clustering via cell-hash noise --
    cluster_mask = _cell_hash_cluster(
        sx, sy, sz, cfg.noise_scale, cfg.cluster_threshold, seed,
    )

    # -- 5. Replaceable block mask --
    block_mask = replace_mask

    # -- 6. Biome mask --
    if cfg.biomes is not None and biome_map is not None:
        biome_ok = np.zeros((sx, sz), dtype=bool)
        for biome in cfg.biomes:
            biome_ok |= biome_map == int(biome)
        # Broadcast (sx, sz) -> (sx, sy, sz)
        biome_mask_3d = np.broadcast_to(
            biome_ok[:, np.newaxis, :], (sx, sy, sz)
        )
    else:
        biome_mask_3d = True  # type: ignore[assignment]

    # -- 7. Combine all masks --
    final_mask = prob_mask & cluster_mask & block_mask & biome_mask_3d

    # -- 8. Air exposure skip --
    if cfg.air_exposure_skip > 0 and np.any(final_mask):
        air_exposed = _check_air_exposure(world_blocks, size)
        skip_rng = np.random.default_rng(seed + 999)
        skip_roll = skip_rng.random((sx, sy, sz))
        skip_mask = air_exposed & (skip_roll < cfg.air_exposure_skip)
        final_mask = final_mask & ~skip_mask

    # -- 9. Place blocks --
    world_blocks[final_mask] = np.int8(cfg.block_type)


def _check_air_exposure(
    world_blocks: np.ndarray,
    size: tuple[int, int, int],
) -> np.ndarray:
    """Return a boolean mask of voxels adjacent to at least one air block.

    Checks all 6 cardinal neighbors using array shifting (vectorized).
    """
    sx, sy, sz = size
    exposed = np.zeros((sx, sy, sz), dtype=bool)

    air = world_blocks == BlockType.AIR

    # +x / -x
    exposed[:-1, :, :] |= air[1:, :, :]
    exposed[1:, :, :] |= air[:-1, :, :]
    # +y / -y
    exposed[:, :-1, :] |= air[:, 1:, :]
    exposed[:, 1:, :] |= air[:, :-1, :]
    # +z / -z
    exposed[:, :, :-1] |= air[:, :, 1:]
    exposed[:, :, 1:] |= air[:, :, :-1]

    return exposed
