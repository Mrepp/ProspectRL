"""Pre-process Minecraft world saves into cached numpy arrays for training.

Reads real MC worlds via Amulet Core, converts block/biome data to
ProspectRL's internal format, and saves as .npz files that
``RealChunkWorld`` can load without any Amulet dependency at training time.

Supports multi-world extraction, non-overlapping stride, biome-aware
stratified sampling, incremental caching, and biome diversity reporting.

Re-running on the same output directory is **incremental**: a manifest
tracks which ``(world_path, cx, cz)`` regions have already been cached,
so only new or previously-missing regions are extracted.  This lets you
explore more of a world in-game and re-run to pick up the new chunks.

Usage::

    # Extract from two worlds with analysis
    python -m prospect_rl.tools.cache_chunks \\
        --world-path data/worlds/1 data/worlds/2 \\
        --output-dir data/chunk_cache/combined \\
        --chunk-size 64 384 64 \\
        --y-min -64 --y-max 320 \\
        --samples 9999 \\
        --analyze

    # Re-run after exploring more (only new regions added)
    python -m prospect_rl.tools.cache_chunks \\
        --world-path data/worlds/1 data/worlds/2 \\
        --output-dir data/chunk_cache/combined \\
        --samples 9999 --analyze
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "_manifest.json"


def _require_amulet():
    """Import amulet and raise a clear error if not installed."""
    try:
        import amulet  # noqa: F401
        return amulet
    except ImportError:
        print(
            "ERROR: amulet-core is required for chunk caching.\n"
            "Install with: pip install 'prospect-rl[worldimport]'",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Manifest helpers (incremental caching)
# ---------------------------------------------------------------------------

def _load_manifest(out_path: Path) -> dict:
    """Load the cache manifest, or return an empty one.

    Manifest structure::

        {
            "next_id": 42,
            "regions": {
                "<world_abspath>|<cx>|<cz>": "00012.npz",
                ...
            }
        }
    """
    mf = out_path / _MANIFEST_NAME
    if mf.exists():
        try:
            data = json.loads(mf.read_text())
            # Validate
            if "next_id" in data and "regions" in data:
                return data
        except Exception:
            logger.warning("Corrupt manifest, starting fresh")
    return {"next_id": 0, "regions": {}}


def _save_manifest(out_path: Path, manifest: dict) -> None:
    mf = out_path / _MANIFEST_NAME
    mf.write_text(json.dumps(manifest, indent=1, sort_keys=True))


def _region_key(world_path: str, cx: int, cz: int) -> str:
    """Canonical key for a region in the manifest."""
    return f"{Path(world_path).resolve()}|{cx}|{cz}"


# ---------------------------------------------------------------------------
# Biome helpers
# ---------------------------------------------------------------------------

def _classify_region_biome(biome_map: np.ndarray) -> int:
    """Return the dominant BiomeType from a (sx, sz) biome_map.

    Tie-breaks toward rarer biomes (higher enum values).
    """
    unique, counts = np.unique(biome_map, return_counts=True)
    order = np.lexsort((-unique, -counts))
    return int(unique[order[0]])


def _probe_biome_multi_y(
    level,
    cx: int,
    cz: int,
    mc_y_min: int,
    mc_y_max: int,
    dimension: str,
) -> int | None:
    """Quick biome probe at multiple Y-levels to catch cave biomes.

    Amulet biomes use 4-block resolution ``(4, inf, 4)`` with string
    palette entries (``universal_minecraft:`` prefix).

    Returns the rarest BiomeType found, or None if unreadable.
    """
    from amulet.api.errors import ChunkDoesNotExist, ChunkLoadError
    from prospect_rl.env.world.mc_block_mapping import (
        mc_biome_to_biometype,
    )

    mc_y_range = mc_y_max - mc_y_min
    probe_ys = [
        mc_y_min + mc_y_range // 4,
        mc_y_min + mc_y_range // 2,
        mc_y_min + 3 * mc_y_range // 4,
    ]

    biomes_found: set[int] = set()
    try:
        chunk = level.get_chunk(cx, cz, dimension)
        biome_palette = chunk.biome_palette
        for biome_y in probe_ys:
            for bx in (0, 1, 2, 3):
                for bz in (0, 1, 2, 3):
                    try:
                        biome_id = chunk.biomes[bx, biome_y, bz]
                        bname = biome_palette[biome_id]
                        if isinstance(bname, str):
                            bt = mc_biome_to_biometype(bname)
                        else:
                            bt = mc_biome_to_biometype(
                                bname.namespaced_name,
                            )
                        biomes_found.add(bt)
                    except Exception:
                        pass
    except (ChunkDoesNotExist, ChunkLoadError):
        return None

    if not biomes_found:
        return None
    return max(biomes_found)


def _stratified_sample(
    valid_starts: list[tuple[int, int]],
    biome_labels: dict[tuple[int, int], int],
    num_samples: int,
    rng: np.random.Generator,
    min_per_biome: int = 0,
) -> list[tuple[int, int]]:
    """Select starts with biome diversity guarantees."""
    buckets: dict[int, list[tuple[int, int]]] = {}
    for start in valid_starts:
        biome = biome_labels.get(start)
        if biome is not None:
            buckets.setdefault(biome, []).append(start)

    num_biomes = len(buckets)
    if num_biomes == 0:
        return []

    if min_per_biome <= 0:
        min_per_biome = max(5, num_samples // (num_biomes * 2))

    selected: list[tuple[int, int]] = []
    remaining_budget = num_samples

    for biome_val in sorted(buckets.keys(), reverse=True):
        bucket = buckets[biome_val]
        quota = min(min_per_biome, len(bucket), remaining_budget)
        if quota > 0:
            indices = rng.choice(
                len(bucket), size=quota, replace=False,
            )
            selected.extend(bucket[i] for i in indices)
            remaining_budget -= quota

    if remaining_budget > 0:
        selected_set = set(selected)
        pool = [s for s in valid_starts if s not in selected_set]
        if pool:
            take = min(remaining_budget, len(pool))
            indices = rng.choice(len(pool), size=take, replace=False)
            selected.extend(pool[i] for i in indices)

    return selected


def _print_biome_diversity_report(
    biome_counts: dict[int, int],
) -> None:
    """Print biome diversity analysis with warnings."""
    from prospect_rl.config import BiomeType

    total = sum(biome_counts.values())
    if total == 0:
        print("WARNING: No regions cached.")
        return

    num_bt = len(BiomeType)
    expected_pct = 100.0 / num_bt

    print(f"\n{'='*60}")
    print(f"  Biome Diversity Report ({total} total regions)")
    print(f"{'='*60}\n")

    for biome in BiomeType:
        count = biome_counts.get(int(biome), 0)
        pct = count / total * 100
        bar = "#" * int(pct)
        status = ""
        if count == 0:
            status = "  << ABSENT"
        elif pct < expected_pct * 0.25:
            status = "  << SEVERELY UNDERREPRESENTED"
        elif pct < expected_pct * 0.5:
            status = "  << UNDERREPRESENTED"
        print(
            f"  {biome.name:>20s}: "
            f"{count:5d} ({pct:5.1f}%) {bar}{status}"
        )

    present = sum(
        1 for b in BiomeType
        if biome_counts.get(int(b), 0) > 0
    )
    print(f"\n  Biome coverage: {present}/{num_bt} types present")

    if present < num_bt:
        missing = [
            b.name for b in BiomeType
            if biome_counts.get(int(b), 0) == 0
        ]
        print(f"  Missing biomes: {', '.join(missing)}")
        print(
            "  NOTE: These biomes may not exist "
            "in the provided world saves."
        )
    print()


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------

def _get_generated_chunks(
    level,
    dimension: str = "minecraft:overworld",
    max_scan: int = 50_000,
) -> list[tuple[int, int]]:
    """Return (cx, cz) for all chunks that exist in *level*."""
    chunks = []
    for cx, cz in level.all_chunk_coords(dimension):
        chunks.append((cx, cz))
        if len(chunks) >= max_scan:
            break
    return chunks


def _chunk_is_populated(level, cx, cz, dimension) -> bool:
    """Return True if a chunk has real block data (not all air).

    Amulet loads unvisited chunks as palette-size-1 (air only).
    """
    from amulet.api.errors import ChunkDoesNotExist, ChunkLoadError
    try:
        chunk = level.get_chunk(cx, cz, dimension)
        return len(chunk.block_palette) > 1
    except (ChunkDoesNotExist, ChunkLoadError):
        return False


def _build_palette_lut(chunk, default: int) -> np.ndarray:
    """Build a LUT mapping chunk palette indices → BlockType ints.

    Handles ``universal_minecraft:`` prefix via
    ``mc_block_to_blocktype``.
    """
    from prospect_rl.env.world.mc_block_mapping import (
        mc_block_to_blocktype,
    )

    palette_size = len(chunk.block_palette)
    lut = np.full(palette_size, default, dtype=np.int8)
    for idx in range(palette_size):
        try:
            name = chunk.block_palette[idx].namespaced_name
            lut[idx] = mc_block_to_blocktype(name)
        except Exception:
            pass
    return lut


def _extract_region(
    level,
    cx_start: int,
    cz_start: int,
    size_x: int,
    size_y: int,
    size_z: int,
    mc_y_min: int,
    mc_y_max: int,
    dimension: str = "minecraft:overworld",
    min_solid_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract blocks and biomes from the MC world.

    Returns ``(blocks, biome_map)`` or **None** if any required chunk
    is missing, if every chunk in the region is unpopulated
    (all-air / never visited in-game), or if the extracted region has
    less than *min_solid_frac* non-air blocks (e.g. ocean columns
    whose palette contains water but maps to AIR).
    """
    from prospect_rl.env.world.mc_block_mapping import (
        DEFAULT_BIOME,
        DEFAULT_BLOCKTYPE,
        mc_biome_to_biometype,
    )
    from amulet.api.errors import ChunkDoesNotExist, ChunkLoadError

    mc_y_range = mc_y_max - mc_y_min

    if mc_y_range != size_y:
        logger.warning(
            "MC Y range (%d to %d = %d) != sim Y (%d). "
            "Nearest-neighbor resampling.",
            mc_y_min, mc_y_max, mc_y_range, size_y,
        )

    chunks_x = (size_x + 15) // 16
    chunks_z = (size_z + 15) // 16

    # --- Check that all chunks exist AND at least one is populated ---
    any_populated = False
    for dcx in range(chunks_x):
        for dcz in range(chunks_z):
            cx = cx_start + dcx
            cz = cz_start + dcz
            try:
                chunk = level.get_chunk(cx, cz, dimension)
            except (ChunkDoesNotExist, ChunkLoadError):
                return None
            if len(chunk.block_palette) > 1:
                any_populated = True

    if not any_populated:
        return None  # entire region is ungenerated air

    # --- Allocate output ---
    blocks = np.full(
        (size_x, size_y, size_z), DEFAULT_BLOCKTYPE, dtype=np.int8,
    )
    biome_map = np.full(
        (size_x, size_z), DEFAULT_BIOME, dtype=np.int8,
    )

    # Y mapping: sim_y → absolute mc_y
    if mc_y_range == size_y:
        mc_ys = np.arange(mc_y_min, mc_y_max)
    else:
        offsets = np.round(
            np.linspace(0, mc_y_range - 1, size_y),
        ).astype(int)
        mc_ys = mc_y_min + offsets

    # Biome probes at useful depths (not mid-sky)
    biome_probe_ys = [
        max(mc_y_min, -48),   # cave biomes
        max(mc_y_min, 0),     # surface
        max(mc_y_min, 48),    # above surface
    ]

    # --- Fill block & biome data ---
    for dcx in range(chunks_x):
        for dcz in range(chunks_z):
            try:
                chunk = level.get_chunk(
                    cx_start + dcx, cz_start + dcz,
                    dimension,
                )
            except (ChunkDoesNotExist, ChunkLoadError):
                continue

            palette_lut = _build_palette_lut(
                chunk, DEFAULT_BLOCKTYPE,
            )

            sim_x0 = dcx * 16
            sim_z0 = dcz * 16
            x_end = min(16, size_x - sim_x0)
            z_end = min(16, size_z - sim_z0)
            if x_end <= 0 or z_end <= 0:
                continue

            # Blocks (absolute Y coords)
            for lx in range(x_end):
                for lz in range(z_end):
                    for sim_y, mc_y in enumerate(mc_ys):
                        try:
                            rid = chunk.blocks[
                                lx, int(mc_y), lz
                            ]
                            bt = (
                                palette_lut[rid]
                                if rid < len(palette_lut)
                                else DEFAULT_BLOCKTYPE
                            )
                            blocks[
                                sim_x0 + lx, sim_y,
                                sim_z0 + lz,
                            ] = bt
                        except Exception:
                            pass

            # Biomes (4-block resolution, multi-Y probe)
            try:
                bp = chunk.biome_palette
                for lx in range(x_end):
                    bx = min(lx // 4, 3)
                    for lz in range(z_end):
                        bz = min(lz // 4, 3)
                        best = DEFAULT_BIOME
                        for py in biome_probe_ys:
                            try:
                                bid = chunk.biomes[
                                    bx, py, bz
                                ]
                                bn = bp[bid]
                                if isinstance(bn, str):
                                    bt = mc_biome_to_biometype(
                                        bn,
                                    )
                                else:
                                    bt = mc_biome_to_biometype(
                                        bn.namespaced_name,
                                    )
                                if bt > best:
                                    best = bt
                            except Exception:
                                pass
                        biome_map[
                            sim_x0 + lx,
                            sim_z0 + lz,
                        ] = best
            except Exception:
                pass

    # Post-extraction validation: reject regions that are mostly air.
    # Ocean/sky regions pass the palette check (water → AIR) but have
    # no useful solid content for mining training.
    solid = int(np.sum(blocks != 0))  # 0 == BlockType.AIR
    solid_frac = solid / blocks.size
    if solid_frac < min_solid_frac:
        logger.debug(
            "Region cx=%d cz=%d only %.1f%% solid — skipping",
            cx_start, cz_start, solid_frac * 100,
        )
        return None

    return blocks, biome_map


# ---------------------------------------------------------------------------
# Main caching function
# ---------------------------------------------------------------------------

def cache_chunks(
    world_path: str,
    output_dir: str,
    chunk_size: tuple[int, int, int] = (64, 384, 64),
    mc_y_min: int = -64,
    mc_y_max: int = 320,
    num_samples: int = 500,
    seed: int = 42,
    dimension: str = "minecraft:overworld",
    *,
    biome_aware: bool = False,
    min_per_biome: int = 0,
    stride: int = 0,
    world_index: int = 0,
    manifest: dict | None = None,
) -> tuple[int, dict[int, int]]:
    """Extract and cache chunk regions from a MC world.

    *manifest* is the shared manifest dict (loaded once in ``main``).
    Already-cached regions are skipped automatically.

    Returns ``(num_cached, biome_counts)``.
    """
    amulet = _require_amulet()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if manifest is None:
        manifest = _load_manifest(out_path)

    logger.info("Opening world: %s", world_path)
    level = amulet.load_level(world_path)

    biome_counts: Counter[int] = Counter()

    try:
        all_chunks = _get_generated_chunks(level, dimension)
        if not all_chunks:
            logger.error("No chunks found in %s", world_path)
            return 0, dict(biome_counts)

        logger.info("Found %d generated chunks", len(all_chunks))

        rng = np.random.default_rng(seed)
        size_x, size_y, size_z = chunk_size

        chunks_per_x = (size_x + 15) // 16
        chunks_per_z = (size_z + 15) // 16

        # Valid starting positions
        chunk_set = set(all_chunks)
        valid_starts = []
        for cx, cz in all_chunks:
            ok = True
            for dcx in range(chunks_per_x):
                for dcz in range(chunks_per_z):
                    if (cx + dcx, cz + dcz) not in chunk_set:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid_starts.append((cx, cz))

        if not valid_starts:
            logger.error(
                "No valid starts for %dx%d chunk regions",
                chunks_per_x, chunks_per_z,
            )
            return 0, dict(biome_counts)

        logger.info(
            "%d valid region starts for %dx%d chunk regions",
            len(valid_starts), chunks_per_x, chunks_per_z,
        )

        # Non-overlapping stride
        stride_x = stride if stride > 0 else chunks_per_x
        stride_z = stride if stride > 0 else chunks_per_z

        min_cx = min(s[0] for s in valid_starts)
        min_cz = min(s[1] for s in valid_starts)
        strided = [
            (cx, cz) for cx, cz in valid_starts
            if (cx - min_cx) % stride_x == 0
            and (cz - min_cz) % stride_z == 0
        ]
        if strided:
            logger.info(
                "Stride %dx%d: %d -> %d non-overlapping",
                stride_x, stride_z,
                len(valid_starts), len(strided),
            )
            valid_starts = strided
        else:
            logger.warning(
                "Stride produced 0, using all %d",
                len(valid_starts),
            )

        # --- Filter out already-cached regions ---
        wp_resolved = str(Path(world_path).resolve())
        uncached = []
        already = 0
        for cx, cz in valid_starts:
            key = _region_key(world_path, cx, cz)
            if key in manifest["regions"]:
                # Check if file still exists on disk
                fname = manifest["regions"][key]
                if (out_path / fname).exists():
                    already += 1
                    continue
            uncached.append((cx, cz))

        if already:
            logger.info(
                "Skipping %d already-cached regions "
                "(%d new candidates)",
                already, len(uncached),
            )
        valid_starts = uncached

        if not valid_starts:
            logger.info("Nothing new to cache for %s", world_path)
            return 0, dict(biome_counts)

        # Biome-aware stratified sampling
        if biome_aware:
            logger.info(
                "Surveying biomes at %d starts...",
                len(valid_starts),
            )
            biome_labels: dict[tuple[int, int], int] = {}
            for cx, cz in valid_starts:
                center_cx = cx + chunks_per_x // 2
                center_cz = cz + chunks_per_z // 2
                label = _probe_biome_multi_y(
                    level, center_cx, center_cz,
                    mc_y_min, mc_y_max, dimension,
                )
                if label is not None:
                    biome_labels[(cx, cz)] = label

            survey: Counter[int] = Counter(biome_labels.values())
            from prospect_rl.config import BiomeType
            for biome in BiomeType:
                cnt = survey.get(int(biome), 0)
                logger.info(
                    "  Survey: %s = %d", biome.name, cnt,
                )

            selected = _stratified_sample(
                valid_starts, biome_labels,
                num_samples, rng, min_per_biome,
            )
        else:
            n = min(num_samples, len(valid_starts))
            indices = rng.choice(
                len(valid_starts), size=n, replace=False,
            )
            selected = [valid_starts[i] for i in indices]

        logger.info("Extracting %d regions...", len(selected))

        next_id = manifest["next_id"]
        cached = 0
        skipped_empty = 0

        for i, (cx, cz) in enumerate(selected):
            result = _extract_region(
                level, cx, cz,
                size_x, size_y, size_z,
                mc_y_min, mc_y_max,
                dimension,
            )

            if result is None:
                skipped_empty += 1
                continue

            blocks, biome_map = result

            dominant = _classify_region_biome(biome_map)
            biome_counts[dominant] += 1

            fname = f"{next_id:05d}.npz"
            np.savez_compressed(
                out_path / fname,
                blocks=blocks,
                biome_map=biome_map,
                metadata=np.array([
                    size_x, size_y, size_z,
                    mc_y_min, mc_y_max, world_index,
                ]),
            )

            # Update manifest
            key = _region_key(world_path, cx, cz)
            manifest["regions"][key] = fname
            next_id += 1
            cached += 1

            if (i + 1) % 50 == 0:
                logger.info(
                    "Cached %d/%d (skipped %d empty)...",
                    cached, len(selected), skipped_empty,
                )

        manifest["next_id"] = next_id

        if skipped_empty:
            logger.info(
                "Skipped %d ungenerated/empty regions",
                skipped_empty,
            )
        logger.info(
            "Done. Cached %d regions to %s", cached, output_dir,
        )
        return cached, dict(biome_counts)

    finally:
        level.close()


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Cache Minecraft world chunks as numpy arrays "
            "for training."
        ),
    )
    parser.add_argument(
        "--world-path", required=True, nargs="+",
        help=(
            "Path(s) to MC world directories (level.dat). "
            "Multiple worlds extracted into the same output."
        ),
    )
    parser.add_argument(
        "--output-dir", default="data/chunk_cache/default",
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--chunk-size", nargs=3, type=int,
        default=[64, 384, 64],
        metavar=("X", "Y", "Z"),
        help="Sim grid dimensions (default: 64 384 64)",
    )
    parser.add_argument(
        "--y-min", type=int, default=-64,
        help="MC Y minimum (default: -64)",
    )
    parser.add_argument(
        "--y-max", type=int, default=None,
        help="MC Y maximum (default: y_min + chunk_size_y)",
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Max regions per world (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dimension", default="minecraft:overworld",
        help="MC dimension (default: minecraft:overworld)",
    )
    parser.add_argument(
        "--biome-aware", action="store_true",
        help="Stratified sampling for biome diversity.",
    )
    parser.add_argument(
        "--min-per-biome", type=int, default=0,
        help="Min regions per biome (0 = auto).",
    )
    parser.add_argument(
        "--stride", type=int, default=0,
        help="Spacing in chunks (0 = auto non-overlapping).",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run analysis after caching.",
    )

    args = parser.parse_args()

    y_max = args.y_max
    if y_max is None:
        y_max = args.y_min + args.chunk_size[1]
        logger.info(
            "--y-max %d (= y_min + size_y) for 1:1 mapping",
            y_max,
        )

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_path)

    total_cached = 0
    combined_biome: Counter[int] = Counter()

    for wi, wp in enumerate(args.world_path):
        logger.info(
            "=== World %d/%d: %s ===",
            wi + 1, len(args.world_path), wp,
        )

        cached, biome_counts = cache_chunks(
            world_path=wp,
            output_dir=args.output_dir,
            chunk_size=tuple(args.chunk_size),
            mc_y_min=args.y_min,
            mc_y_max=y_max,
            num_samples=args.samples,
            seed=args.seed,
            dimension=args.dimension,
            biome_aware=args.biome_aware,
            min_per_biome=args.min_per_biome,
            stride=args.stride,
            world_index=wi,
            manifest=manifest,
        )

        total_cached += cached
        combined_biome.update(biome_counts)

    # Persist manifest after all worlds
    _save_manifest(out_path, manifest)
    logger.info("Total cached across all worlds: %d", total_cached)

    if args.analyze:
        from prospect_rl.tools.analyze_chunks import (
            analyze_cache,
            print_stats,
        )
        stats = analyze_cache(args.output_dir)
        if stats:
            print_stats(stats)
        _print_biome_diversity_report(dict(combined_biome))

    if total_cached == 0 and manifest["next_id"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
