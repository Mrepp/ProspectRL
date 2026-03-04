"""Analyze cached chunk data and report statistics for sim calibration.

Reads ``.npz`` files produced by ``cache_chunks.py`` and reports
ore densities, biome distributions, air fractions, and block type
breakdowns.  Use the output to tune procedural generation parameters
in ``OreSpawnConfig``.

Usage::

    python -m prospect_rl.tools.analyze_chunks \\
        --cache-dir data/chunk_cache/my_world
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from prospect_rl.config import (
    NUM_ORE_TYPES,
    ORE_TYPES,
    BiomeType,
    BlockType,
)

logger = logging.getLogger(__name__)

# Human-readable names
_ORE_NAMES = ["coal", "iron", "gold", "diamond", "redstone", "emerald", "lapis", "copper"]
_BIOME_NAMES = {int(b): b.name for b in BiomeType}
_BLOCK_NAMES = {int(b): b.name for b in BlockType}


def analyze_cache(cache_dir: str) -> dict:
    """Analyze all cached chunks and return statistics dict."""
    cache_path = Path(cache_dir)
    files = sorted(cache_path.glob("*.npz"))
    if not files:
        logger.error("No .npz files found in %s", cache_dir)
        return {}

    logger.info("Analyzing %d cached chunks in %s", len(files), cache_dir)

    # Accumulators
    total_blocks = 0
    block_counts: Counter = Counter()
    ore_counts = np.zeros(NUM_ORE_TYPES, dtype=np.int64)
    biome_counts: Counter = Counter()
    region_biome_counts: Counter = Counter()
    air_by_y: dict[int, int] = {}
    total_by_y: dict[int, int] = {}
    num_chunks = 0

    for filepath in files:
        data = np.load(filepath, allow_pickle=False)
        blocks = data["blocks"]  # (sx, sy, sz) int8
        biome_map = data["biome_map"]  # (sx, sz) int8

        sx, sy, sz = blocks.shape
        vol = sx * sy * sz
        total_blocks += vol
        num_chunks += 1

        # Block type counts
        for bt in BlockType:
            count = int(np.sum(blocks == int(bt)))
            block_counts[int(bt)] += count

        # Ore counts
        for i, ore_bt in enumerate(ORE_TYPES):
            ore_counts[i] += int(np.sum(blocks == int(ore_bt)))

        # Biome distribution (per XZ cell)
        for b_val in np.unique(biome_map):
            biome_counts[int(b_val)] += int(np.sum(biome_map == b_val))

        # Per-region dominant biome
        unique, counts = np.unique(biome_map, return_counts=True)
        order = np.lexsort((-unique, -counts))
        dominant = int(unique[order[0]])
        region_biome_counts[dominant] += 1

        # Air fraction by Y-level
        for y in range(sy):
            y_slice = blocks[:, y, :]
            air_count = int(np.sum(y_slice == int(BlockType.AIR)))
            air_by_y[y] = air_by_y.get(y, 0) + air_count
            total_by_y[y] = total_by_y.get(y, 0) + (sx * sz)

    # Compute stats
    stats = {
        "num_chunks": num_chunks,
        "total_blocks": total_blocks,
        "ore_density_per_1000": {},
        "block_distribution_pct": {},
        "biome_distribution_pct": {},
        "region_biome_distribution": {},
        "air_fraction_by_y": {},
    }

    # Ore density per 1000 blocks
    for i, name in enumerate(_ORE_NAMES):
        density = ore_counts[i] / total_blocks * 1000 if total_blocks > 0 else 0
        stats["ore_density_per_1000"][name] = round(density, 3)

    # Block distribution
    for bt_int, count in block_counts.most_common():
        name = _BLOCK_NAMES.get(bt_int, f"unknown_{bt_int}")
        pct = count / total_blocks * 100 if total_blocks > 0 else 0
        stats["block_distribution_pct"][name] = round(pct, 2)

    # Biome distribution
    total_biome_cells = sum(biome_counts.values())
    for b_int, count in biome_counts.most_common():
        name = _BIOME_NAMES.get(b_int, f"unknown_{b_int}")
        pct = count / total_biome_cells * 100 if total_biome_cells > 0 else 0
        stats["biome_distribution_pct"][name] = round(pct, 2)

    # Region-level dominant biome distribution
    for b_int, count in region_biome_counts.most_common():
        name = _BIOME_NAMES.get(b_int, f"unknown_{b_int}")
        stats["region_biome_distribution"][name] = count

    # Air fraction by Y
    for y in sorted(air_by_y.keys()):
        frac = air_by_y[y] / total_by_y[y] if total_by_y.get(y, 0) > 0 else 0
        stats["air_fraction_by_y"][y] = round(frac, 4)

    return stats


def print_stats(stats: dict) -> None:
    """Pretty-print analysis statistics."""
    if not stats:
        return

    print(f"\n{'='*60}")
    print(f"  Chunk Cache Analysis ({stats['num_chunks']} chunks, "
          f"{stats['total_blocks']:,} total blocks)")
    print(f"{'='*60}\n")

    print("  Ore density (per 1000 blocks):")
    for name, density in stats["ore_density_per_1000"].items():
        bar = "#" * int(density * 5)
        print(f"    {name:>10s}: {density:8.3f}  {bar}")

    print(f"\n  Block distribution:")
    for name, pct in stats["block_distribution_pct"].items():
        if pct >= 0.01:
            print(f"    {name:>15s}: {pct:6.2f}%")

    print(f"\n  Biome distribution (per XZ cell):")
    for name, pct in stats["biome_distribution_pct"].items():
        print(f"    {name:>20s}: {pct:6.2f}%")

    region_biomes = stats.get("region_biome_distribution", {})
    if region_biomes:
        total_regions = sum(region_biomes.values())
        print(f"\n  Region-level dominant biome ({total_regions} regions):")
        for name, count in region_biomes.items():
            pct = count / total_regions * 100
            bar = "#" * int(pct / 2)
            print(f"    {name:>20s}: {count:5d} ({pct:5.1f}%) {bar}")

    air_by_y = stats["air_fraction_by_y"]
    if air_by_y:
        print(f"\n  Air fraction by Y-level (cave profile):")
        max_y = max(air_by_y.keys())
        step = max(1, max_y // 20)
        for y in range(0, max_y + 1, step):
            frac = air_by_y.get(y, 0.0)
            bar = "#" * int(frac * 50)
            print(f"    Y={y:3d}: {frac:.4f}  {bar}")

    print()


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Analyze cached Minecraft chunk data for sim calibration.",
    )
    parser.add_argument(
        "--cache-dir", default="data/chunk_cache/default",
        help="Directory containing .npz cache files",
    )

    args = parser.parse_args()

    stats = analyze_cache(args.cache_dir)
    if stats:
        print_stats(stats)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
