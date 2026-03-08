"""Generate terrain profiles from cached real Minecraft chunks.

Computes three per-(Y-level, biome) profiles and saves as ``.npz``:

1. **Solid fraction** — P(solid | y, biome).
2. **Replaceable fraction** — P(block in ``stone_ore_replaceables`` | y, biome).
   This is the fraction relevant for ore placement: only stone, deepslate,
   granite, diorite, andesite, tuff (and existing ores that occupy those
   positions) are valid ore targets.
3. **Air adjacency** — P(any of 6 neighbours is air | solid block at y, biome).
   Used to model the ``discard_chance_on_air_exposure`` parameter.

Usage::

    python -m prospect_rl.tools.generate_solid_profile \\
        --cache-dir data/chunk_cache/combined \\
        --output data/solid_fraction_profile.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from prospect_rl.config import BlockType, NUM_BIOME_TYPES

logger = logging.getLogger(__name__)

# Block types that are in MC's ``stone_ore_replaceables`` or
# ``deepslate_ore_replaceables`` tags.  Ores themselves are included
# because they occupy positions that *were* replaceable before ore gen.
_REPLACEABLE_BLOCK_VALUES: frozenset[int] = frozenset(
    int(bt) for bt in (
        BlockType.STONE, BlockType.GRANITE, BlockType.DIORITE,
        BlockType.ANDESITE, BlockType.TUFF, BlockType.DEEPSLATE,
        BlockType.COAL_ORE, BlockType.IRON_ORE, BlockType.GOLD_ORE,
        BlockType.DIAMOND_ORE, BlockType.REDSTONE_ORE,
        BlockType.EMERALD_ORE, BlockType.LAPIS_ORE, BlockType.COPPER_ORE,
    )
)


def generate_profiles(cache_dir: str) -> dict[str, np.ndarray]:
    """Compute solid, replaceable, and air-adjacency profiles.

    Returns a dict with keys ``solid_fraction``, ``replaceable_fraction``,
    ``air_adjacency``, ``sample_counts`` — each ``(world_height, NUM_BIOME_TYPES)``.
    Also includes ``num_chunks`` (scalar).
    """
    cache_path = Path(cache_dir)
    files = sorted(cache_path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {cache_dir}")

    # Auto-detect world height
    first = np.load(files[0], allow_pickle=False)
    world_height = first["blocks"].shape[1]
    logger.info(
        "Found %d chunks (world_height=%d) in %s",
        len(files), world_height, cache_dir,
    )

    total_voxels = np.zeros((world_height, NUM_BIOME_TYPES), dtype=np.int64)
    solid_voxels = np.zeros((world_height, NUM_BIOME_TYPES), dtype=np.int64)
    replaceable_voxels = np.zeros((world_height, NUM_BIOME_TYPES), dtype=np.int64)
    # Solid blocks with at least one air neighbour
    air_adj_solid = np.zeros((world_height, NUM_BIOME_TYPES), dtype=np.int64)

    repl_values = np.array(sorted(_REPLACEABLE_BLOCK_VALUES), dtype=np.int8)
    num_chunks = 0

    for fi, filepath in enumerate(files):
        data = np.load(filepath, allow_pickle=False)
        blocks = data["blocks"]  # (sx, sy, sz) int8
        biome_map = data["biome_map"]  # (sx, sz) int8

        if blocks.shape[1] != world_height:
            logger.warning(
                "Skipping %s: height %d != expected %d",
                filepath.name, blocks.shape[1], world_height,
            )
            continue

        num_chunks += 1
        sx, sy, sz = blocks.shape
        is_air = blocks == 0
        is_solid = ~is_air

        # Build replaceable mask (vectorised isin)
        is_replaceable = np.isin(blocks, repl_values)

        # Air-adjacency: solid block with >= 1 air neighbour (6-connected)
        has_air_nbr = np.zeros_like(is_air)
        has_air_nbr[1:, :, :] |= is_air[:-1, :, :]
        has_air_nbr[:-1, :, :] |= is_air[1:, :, :]
        has_air_nbr[:, 1:, :] |= is_air[:, :-1, :]
        has_air_nbr[:, :-1, :] |= is_air[:, 1:, :]
        has_air_nbr[:, :, 1:] |= is_air[:, :, :-1]
        has_air_nbr[:, :, :-1] |= is_air[:, :, 1:]
        solid_with_air = is_solid & has_air_nbr

        for biome_id in np.unique(biome_map):
            biome_id_c = int(np.clip(biome_id, 0, NUM_BIOME_TYPES - 1))
            col_mask = biome_map == biome_id
            xs, zs = np.where(col_mask)
            n_cols = len(xs)
            if n_cols == 0:
                continue

            # Shape: (n_cols, world_height)
            col_blocks_solid = is_solid[xs, :, zs]
            col_blocks_repl = is_replaceable[xs, :, zs]
            col_blocks_air_adj = solid_with_air[xs, :, zs]

            total_voxels[:, biome_id_c] += n_cols
            solid_voxels[:, biome_id_c] += col_blocks_solid.sum(axis=0)
            replaceable_voxels[:, biome_id_c] += col_blocks_repl.sum(axis=0)
            air_adj_solid[:, biome_id_c] += col_blocks_air_adj.sum(axis=0)

        if (fi + 1) % 50 == 0:
            logger.info("  processed %d / %d chunks", fi + 1, len(files))

    logger.info(
        "Finished: %d chunks, %s total voxels",
        num_chunks, f"{total_voxels.sum():,}",
    )

    # Compute fractions
    denom_total = np.maximum(total_voxels.astype(np.float64), 1.0)
    denom_solid = np.maximum(solid_voxels.astype(np.float64), 1.0)

    return {
        "solid_fraction": solid_voxels.astype(np.float64) / denom_total,
        "replaceable_fraction": replaceable_voxels.astype(np.float64) / denom_total,
        "air_adjacency": air_adj_solid.astype(np.float64) / denom_solid,
        "sample_counts": total_voxels,
        "num_chunks": np.array(num_chunks),
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate terrain profiles from cached MC chunks.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/chunk_cache/combined",
        help="Directory containing .npz cache files",
    )
    parser.add_argument(
        "--output",
        default="data/solid_fraction_profile.npz",
        help="Output .npz path (default: data/solid_fraction_profile.npz)",
    )
    args = parser.parse_args()

    profiles = generate_profiles(args.cache_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **profiles)
    logger.info(
        "Saved terrain profiles to %s (shape=%s, %d chunks)",
        out_path,
        profiles["solid_fraction"].shape,
        int(profiles["num_chunks"]),
    )

    # Print summary
    sf = profiles["solid_fraction"]
    rf = profiles["replaceable_fraction"]
    aa = profiles["air_adjacency"]
    world_height = sf.shape[0]

    avg_sf = sf.mean(axis=1)
    avg_rf = rf.mean(axis=1)
    avg_aa = aa.mean(axis=1)

    print(f"\nProfile summary ({int(profiles['num_chunks'])} chunks):")
    print(f"  {'y':>3s} {'MC_Y':>4s}  {'solid%':>6s}  {'repl%':>6s}  {'adjAir%':>7s}")
    for y_idx in range(0, world_height, 20):
        mc_y = -64 + y_idx
        print(
            f"  {y_idx:3d} {mc_y:4d}  "
            f"{avg_sf[y_idx] * 100:5.1f}%  "
            f"{avg_rf[y_idx] * 100:5.1f}%  "
            f"{avg_aa[y_idx] * 100:6.1f}%"
        )


if __name__ == "__main__":
    main()
