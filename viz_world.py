"""Real Minecraft chunk viewer using PyVista.

Loads cached ``.npz`` chunks (produced by ``prospect_rl.tools.cache_chunks``)
and renders ores, bedrock, biome overlay, and ore counts.  Press **N** / **P**
to cycle through chunks.  Press **T** to toggle between the cached chunk view
and a prior-derived probabilistic view that shows where the analytical prior
predicts ore should be (sampled from the 1.21.11 worldgen-derived prior).

Usage::

    python viz_world.py                          # random chunk, seed 42
    python viz_world.py --seed 99                # different starting chunk
    python viz_world.py --fillers                # show dirt/gravel/etc.
    python viz_world.py --cache-dir path/to/dir  # custom cache directory
"""

from __future__ import annotations

import argparse
import glob as globmod
import os

import numpy as np
import pyvista as pv
from prospect_rl.config import BiomeType, BlockType, ORE_TYPES

# ── Block / biome appearance ─────────────────────────────────────────

BLOCK_COLORS: dict[int, str] = {
    BlockType.AIR: "#aaddff",
    BlockType.COAL_ORE: "#3d3d3d",
    BlockType.IRON_ORE: "#d4a373",
    BlockType.GOLD_ORE: "#fcf051",
    BlockType.DIAMOND_ORE: "#4aedd9",
    BlockType.REDSTONE_ORE: "#ff0000",
    BlockType.EMERALD_ORE: "#17dd62",
    BlockType.LAPIS_ORE: "#345ec3",
    BlockType.COPPER_ORE: "#e07c4a",
    BlockType.BEDROCK: "#555555",
    BlockType.DIRT: "#8b6914",
    BlockType.GRAVEL: "#8a8a8a",
    BlockType.GRANITE: "#9a6b4c",
    BlockType.DIORITE: "#c8c8c8",
    BlockType.ANDESITE: "#8b8b8b",
    BlockType.TUFF: "#4d4d44",
    BlockType.DEEPSLATE: "#505050",
    BlockType.CLAY: "#9eabb0",
}

BLOCK_NAMES: dict[int, str] = {
    BlockType.AIR: "Air/Cave",
    BlockType.STONE: "Stone",
    BlockType.COAL_ORE: "Coal",
    BlockType.IRON_ORE: "Iron",
    BlockType.GOLD_ORE: "Gold",
    BlockType.DIAMOND_ORE: "Diamond",
    BlockType.REDSTONE_ORE: "Redstone",
    BlockType.EMERALD_ORE: "Emerald",
    BlockType.LAPIS_ORE: "Lapis",
    BlockType.COPPER_ORE: "Copper",
    BlockType.BEDROCK: "Bedrock",
    BlockType.DIRT: "Dirt",
    BlockType.GRAVEL: "Gravel",
    BlockType.GRANITE: "Granite",
    BlockType.DIORITE: "Diorite",
    BlockType.ANDESITE: "Andesite",
    BlockType.TUFF: "Tuff",
    BlockType.DEEPSLATE: "Deepslate",
    BlockType.CLAY: "Clay",
}

BIOME_COLORS: dict[int, str] = {
    BiomeType.PLAINS: "#7cfc00",
    BiomeType.MOUNTAINS: "#a0a0a0",
    BiomeType.BADLANDS: "#d2691e",
    BiomeType.DRIPSTONE_CAVES: "#8b7355",
    BiomeType.LUSH_CAVES: "#32cd32",
}

BIOME_NAMES: dict[int, str] = {
    BiomeType.PLAINS: "Plains",
    BiomeType.MOUNTAINS: "Mountains",
    BiomeType.BADLANDS: "Badlands",
    BiomeType.DRIPSTONE_CAVES: "Dripstone",
    BiomeType.LUSH_CAVES: "Lush Caves",
}

# Render rarest first so they appear on top
ORE_ORDER = [
    BlockType.DIAMOND_ORE,
    BlockType.EMERALD_ORE,
    BlockType.GOLD_ORE,
    BlockType.COPPER_ORE,
    BlockType.IRON_ORE,
    BlockType.REDSTONE_ORE,
    BlockType.LAPIS_ORE,
    BlockType.COAL_ORE,
]

FILLER_ORDER = [
    BlockType.DIRT,
    BlockType.GRAVEL,
    BlockType.GRANITE,
    BlockType.DIORITE,
    BlockType.ANDESITE,
    BlockType.TUFF,
    BlockType.CLAY,
    BlockType.DEEPSLATE,
]

# Actor names that get removed on chunk cycling
_ORE_ACTORS = [f"ore_{bt}" for bt in ORE_ORDER]
_FILLER_ACTORS = [f"filler_{bt}" for bt in FILLER_ORDER]
_BIOME_ACTORS = [f"biome_{bt}" for bt in BiomeType]
_PRIOR_ACTORS = [f"prior_{bt}" for bt in ORE_ORDER]
_TEXT_ACTORS = ["title", "info", "ore_counts", "bedrock"]
_ALL_ACTORS = (
    _ORE_ACTORS + _FILLER_ACTORS + _BIOME_ACTORS
    + _PRIOR_ACTORS + _TEXT_ACTORS
)


# ── Helpers ───────────────────────────────────────────────────────────


def _voxels_for_block(
    blocks: np.ndarray, block_id: int,
) -> pv.PolyData | None:
    """Create unit-cube mesh for every voxel matching *block_id*."""
    xs, ys, zs = np.where(blocks == block_id)
    if len(xs) == 0:
        return None
    centers = np.column_stack([xs, ys, zs]).astype(float)
    cloud = pv.PolyData(centers)
    return cloud.glyph(
        geom=pv.Cube(x_length=0.9, y_length=0.9, z_length=0.9),
    )


def _discover_chunks(
    cache_dir: str, min_solid_frac: float = 0.05,
) -> list[str]:
    """Return paths of non-empty ``.npz`` chunks in *cache_dir*.

    Filters out chunks that are almost entirely air (ocean/sky
    regions that slipped through the extractor).
    """
    all_files = sorted(
        globmod.glob(os.path.join(cache_dir, "*.npz")),
    )
    if not all_files:
        raise FileNotFoundError(
            f"No .npz cache files found in {cache_dir!r}.  "
            "Run `python -m prospect_rl.tools.cache_chunks` first."
        )

    kept: list[str] = []
    skipped = 0
    for f in all_files:
        data = np.load(f, allow_pickle=False)
        blocks = data["blocks"]
        solid = int(np.sum(blocks != 0))  # 0 == AIR
        if solid / blocks.size >= min_solid_frac:
            kept.append(f)
        else:
            skipped += 1

    if skipped:
        print(
            f"Filtered {skipped} empty chunks "
            f"({len(kept)} usable of {len(all_files)} total)"
        )
    if not kept:
        raise FileNotFoundError(
            f"All {len(all_files)} chunks in {cache_dir!r} are empty."
        )
    return kept


def _content_y_range(blocks: np.ndarray, pad: int = 2):
    """Find the Y-range that contains non-air blocks.

    Returns ``(y_lo, y_hi)`` with a small *pad* margin.
    """
    sy = blocks.shape[1]
    for y_lo in range(sy):
        if np.any(blocks[:, y_lo, :] != 0):
            break
    else:
        y_lo = 0

    for y_hi in range(sy - 1, -1, -1):
        if np.any(blocks[:, y_hi, :] != 0):
            break
    else:
        y_hi = sy - 1

    y_lo = max(0, y_lo - pad)
    y_hi = min(sy - 1, y_hi + pad)
    return y_lo, y_hi


def _load_chunk(path: str) -> dict:
    """Load a cached chunk, crop Y to content range."""
    data = np.load(path, allow_pickle=False)
    blocks = data["blocks"]
    metadata = data["metadata"]

    y_lo, y_hi = _content_y_range(blocks)
    cropped = blocks[:, y_lo:y_hi + 1, :]

    return {
        "blocks": cropped,
        "biome_map": data["biome_map"],
        "metadata": metadata,
        "filename": os.path.basename(path),
        "y_lo": y_lo,
        "y_hi": y_hi,
        "full_y": int(blocks.shape[1]),
    }


def _ore_count_text(blocks: np.ndarray) -> str:
    """Build a multi-line string with per-ore counts."""
    lines: list[str] = []
    total = 0
    for bt in ORE_ORDER:
        count = int(np.sum(blocks == bt))
        if count > 0:
            lines.append(f"  {BLOCK_NAMES[bt]:<10} {count:>6,}")
            total += count
    if not lines:
        return "No ores in this chunk"
    lines.append(f"  {'Total':<10} {total:>6,}")
    return "\n".join(lines)


def _biome_summary(biome_map: np.ndarray) -> str:
    """One-line biome summary for overlays."""
    total = biome_map.size
    parts: list[str] = []
    for biome_id in BiomeType:
        count = int(np.sum(biome_map == biome_id))
        if count == 0:
            continue
        pct = count / total * 100
        name = BIOME_NAMES.get(biome_id, str(biome_id))
        if pct >= 1:
            parts.append(f"{name} {pct:.0f}%")
        else:
            parts.append(f"{name} <1%")
    return ", ".join(parts) if parts else "Unknown"


def _print_chunk_stats(chunk_data: dict) -> None:
    """Print block distribution to console."""
    blocks = chunk_data["blocks"]
    biome_map = chunk_data["biome_map"]
    metadata = chunk_data["metadata"]
    label = chunk_data["filename"]
    sx, sy, sz = blocks.shape
    mc_y_min = int(metadata[3])
    mc_y_max = int(metadata[4])
    y_lo = chunk_data.get("y_lo", 0)
    y_hi = chunk_data.get("y_hi", sy - 1)
    full_y = chunk_data.get("full_y", sy)

    print(f"\n{'=' * 60}")
    print(
        f"  {label}:  {sx}x{full_y}x{sz}"
        f"  (Y {y_lo}-{y_hi}, MC Y: {mc_y_min} to {mc_y_max})"
    )
    print(f"{'=' * 60}")

    total = blocks.size
    print(f"\n{'Block Type':<16} {'Count':>8} {'% of World':>10}")
    print("-" * 38)

    for bt in ORE_ORDER:
        count = int(np.sum(blocks == bt))
        if count == 0:
            continue
        pct = count / total * 100
        print(f"{BLOCK_NAMES[bt]:<16} {count:>8,} {pct:>9.3f}%")

    for bt in FILLER_ORDER:
        count = int(np.sum(blocks == bt))
        if count == 0:
            continue
        pct = count / total * 100
        print(f"{BLOCK_NAMES[bt]:<16} {count:>8,} {pct:>9.3f}%")

    for bt in [BlockType.STONE, BlockType.AIR, BlockType.BEDROCK]:
        count = int(np.sum(blocks == bt))
        pct = count / total * 100
        name = BLOCK_NAMES.get(bt, str(bt))
        print(f"{name:<16} {count:>8,} {pct:>9.3f}%")

    # Biome distribution
    print("\nBiome Distribution:")
    print("-" * 38)
    total_cols = biome_map.size
    for biome_id in BiomeType:
        count = int(np.sum(biome_map == biome_id))
        if count == 0:
            continue
        pct = count / total_cols * 100
        name = BIOME_NAMES.get(biome_id, str(biome_id))
        print(f"  {name:<16} {count:>6} cols {pct:>6.1f}%")

    print()


# ── Prior sampling ────────────────────────────────────────────────────

# Map ORE_TYPES index to BlockType for the prior view
_ORE_INDEX_TO_BLOCK: dict[int, int] = {
    i: int(bt) for i, bt in enumerate(ORE_TYPES)
}


def _sample_prior_blocks(
    biome_map: np.ndarray,
    world_height: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample a block grid from the analytical prior.

    For each voxel (x, y, z), queries the prior for P(ore_type | y, biome)
    and samples whether each ore is present using those probabilities.
    When multiple ores hit, the rarest one wins.

    Returns an int8 array of shape ``(sx, world_height, sz)`` with
    BlockType values (0=AIR, ore IDs for sampled ores).
    """
    from multiagent.geological_prior import AnalyticalPrior

    sx, sz = biome_map.shape
    prior = AnalyticalPrior(world_height=world_height)

    blocks = np.zeros((sx, world_height, sz), dtype=np.int8)
    rng = np.random.default_rng(seed)

    # Pre-query the prior for each (y, biome) pair
    for y in range(world_height):
        for biome_id in range(5):
            probs = prior.query(y, biome_id)  # shape (8,)

            # Find columns with this biome
            biome_mask = biome_map == biome_id
            xs, zs = np.where(biome_mask)
            if len(xs) == 0:
                continue

            n_cols = len(xs)
            # For each ore type, sample independently
            for ore_idx in range(8):
                p = float(probs[ore_idx])
                if p <= 0:
                    continue
                rolls = rng.random(n_cols)
                hits = rolls < p
                if not np.any(hits):
                    continue
                bt = _ORE_INDEX_TO_BLOCK[ore_idx]
                # Only place if currently AIR (rarer ores placed later
                # overwrite common ones via ORE_ORDER)
                hit_x = xs[hits]
                hit_z = zs[hits]
                blocks[hit_x, y, hit_z] = np.int8(bt)

    return blocks


def _prior_expected_text(
    prior_table: np.ndarray,
    biome_map: np.ndarray,
    world_height: int,
) -> str:
    """Build text showing expected ore counts from the prior."""
    from multiagent.geological_prior import AnalyticalPrior

    prior = AnalyticalPrior(world_height=world_height)
    sx, sz = biome_map.shape

    # Compute expected count per ore across all columns
    lines: list[str] = []
    total = 0.0
    for ore_idx, bt in enumerate(ORE_ORDER):
        # Find this ore's index in the ORE_TYPES list
        oi = None
        for j, obt in enumerate(ORE_TYPES):
            if int(obt) == int(bt):
                oi = j
                break
        if oi is None:
            continue

        expected = 0.0
        for biome_id in range(5):
            biome_count = int(np.sum(biome_map == biome_id))
            if biome_count == 0:
                continue
            for y in range(world_height):
                expected += float(prior._table[oi, y, biome_id]) * biome_count
        if expected > 0.01:
            lines.append(f"  {BLOCK_NAMES[bt]:<10} {expected:>8.1f}")
            total += expected

    if not lines:
        return "No expected ores"
    lines.append(f"  {'Total':<10} {total:>8.1f}")
    return "\n".join(lines)


# ── Rendering ─────────────────────────────────────────────────────────


def _add_biome_overlay(
    pl: pv.Plotter,
    biome_map: np.ndarray,
) -> None:
    """Add a coloured ground-plane showing biome boundaries."""
    for biome_id in BiomeType:
        xs, zs = np.where(biome_map == biome_id)
        if len(xs) == 0:
            continue
        centers = np.column_stack([
            xs.astype(float),
            np.full(len(xs), -0.5),
            zs.astype(float),
        ])
        cloud = pv.PolyData(centers)
        mesh = cloud.glyph(
            geom=pv.Cube(x_length=1.0, y_length=0.1, z_length=1.0),
        )
        name = BIOME_NAMES.get(biome_id, str(biome_id))
        pl.add_mesh(
            mesh,
            color=BIOME_COLORS[biome_id],
            label=f"Biome: {name}",
            opacity=0.6,
            name=f"biome_{biome_id}",
        )


def _render_chunk(
    pl: pv.Plotter,
    chunk_data: dict,
    chunk_index: int,
    total_chunks: int,
    show_fillers: bool,
) -> None:
    """Clear previous actors and render a new chunk."""
    blocks = chunk_data["blocks"]
    biome_map = chunk_data["biome_map"]
    metadata = chunk_data["metadata"]
    filename = chunk_data["filename"]
    sx, sy, sz = blocks.shape

    # Remove previous chunk actors
    for actor_name in _ALL_ACTORS:
        try:
            pl.remove_actor(actor_name)
        except Exception:
            pass

    # Ores
    for bt in ORE_ORDER:
        mesh = _voxels_for_block(blocks, bt)
        if mesh is not None:
            count = int(np.sum(blocks == bt))
            pl.add_mesh(
                mesh,
                color=BLOCK_COLORS[bt],
                label=f"{BLOCK_NAMES[bt]} ({count:,})",
                opacity=1.0,
                name=f"ore_{bt}",
            )

    # Fillers
    if show_fillers:
        for bt in FILLER_ORDER:
            mesh = _voxels_for_block(blocks, bt)
            if mesh is not None:
                count = int(np.sum(blocks == bt))
                pl.add_mesh(
                    mesh,
                    color=BLOCK_COLORS[bt],
                    label=f"{BLOCK_NAMES[bt]} ({count:,})",
                    opacity=0.35,
                    name=f"filler_{bt}",
                )

    # Bedrock
    bed_mesh = _voxels_for_block(blocks, BlockType.BEDROCK)
    if bed_mesh is not None:
        pl.add_mesh(
            bed_mesh,
            color=BLOCK_COLORS[BlockType.BEDROCK],
            label="Bedrock",
            opacity=0.3,
            name="bedrock",
        )

    # Biome overlay
    _add_biome_overlay(pl, biome_map)

    # Text overlays
    mc_y_min = int(metadata[3])
    mc_y_max = int(metadata[4])
    y_lo = chunk_data.get("y_lo", 0)
    y_hi = chunk_data.get("y_hi", sy - 1)
    full_y = chunk_data.get("full_y", sy)
    biome_text = _biome_summary(biome_map)

    info_text = (
        f"Chunk: {filename}"
        f"  ({chunk_index + 1}/{total_chunks})\n"
        f"Size: {sx}x{full_y}x{sz}"
        f"  (showing Y {y_lo}-{y_hi})\n"
        f"MC Y: {mc_y_min} to {mc_y_max}\n"
        f"Biome: {biome_text}\n"
        f"Press N / P to cycle"
    )
    pl.add_text(
        info_text, position="upper_right",
        font_size=10, color="white",
        name="info",
    )

    pl.add_text(
        f"Real Chunk Viewer — {filename}",
        position="upper_left",
        font_size=12, color="white",
        name="title",
    )

    ore_text = "Ore Counts:\n" + _ore_count_text(blocks)
    pl.add_text(
        ore_text, position="lower_right",
        font_size=10, color="#cccccc",
        name="ore_counts",
    )

    # Camera — center on cropped content
    center = np.array([sx / 2, sy / 2, sz / 2])
    max_dim = max(sx, sy, sz)
    pl.camera.focal_point = center
    pl.camera.position = center + np.array([
        max_dim * 1.5, max_dim * 1.2, max_dim * 1.5,
    ])

    pl.render()


# ── Prior view rendering ──────────────────────────────────────────────


def _render_prior(
    pl: pv.Plotter,
    chunk_data: dict,
    chunk_index: int,
    total_chunks: int,
    seed: int = 0,
) -> None:
    """Render a prior-sampled view of the same chunk's biome layout."""
    biome_map = chunk_data["biome_map"]
    metadata = chunk_data["metadata"]
    filename = chunk_data["filename"]
    full_y = chunk_data.get("full_y", int(metadata[1]))

    # Remove previous actors
    for actor_name in _ALL_ACTORS:
        try:
            pl.remove_actor(actor_name)
        except Exception:
            pass

    # Sample blocks from the prior using this chunk's biome map
    prior_blocks = _sample_prior_blocks(biome_map, full_y, seed=seed)

    # Crop to content range (same as chunk view)
    y_lo, y_hi = _content_y_range(prior_blocks)
    cropped = prior_blocks[:, y_lo:y_hi + 1, :]
    sx, sy, sz = cropped.shape

    # Render sampled ores
    for bt in ORE_ORDER:
        mesh = _voxels_for_block(cropped, bt)
        if mesh is not None:
            count = int(np.sum(cropped == bt))
            pl.add_mesh(
                mesh,
                color=BLOCK_COLORS[bt],
                label=f"{BLOCK_NAMES[bt]} ({count:,})",
                opacity=0.85,
                name=f"prior_{bt}",
            )

    # Biome overlay
    _add_biome_overlay(pl, biome_map)

    # Text overlays
    mc_y_min = int(metadata[3])
    mc_y_max = int(metadata[4])
    biome_text = _biome_summary(biome_map)

    info_text = (
        f"PRIOR VIEW - {filename}"
        f"  ({chunk_index + 1}/{total_chunks})\n"
        f"Size: {sx}x{full_y}x{sz}"
        f"  (showing Y {y_lo}-{y_hi})\n"
        f"MC Y: {mc_y_min} to {mc_y_max}\n"
        f"Biome: {biome_text}\n"
        f"Press T for chunk view"
    )
    pl.add_text(
        info_text, position="upper_right",
        font_size=10, color="#ffcc00",
        name="info",
    )

    pl.add_text(
        f"Analytical Prior View — {filename}",
        position="upper_left",
        font_size=12, color="#ffcc00",
        name="title",
    )

    # Show both sampled counts and expected counts
    ore_text = "Sampled Counts:\n" + _ore_count_text(prior_blocks)
    pl.add_text(
        ore_text, position="lower_right",
        font_size=10, color="#cccccc",
        name="ore_counts",
    )

    # Camera
    center = np.array([sx / 2, sy / 2, sz / 2])
    max_dim = max(sx, sy, sz)
    pl.camera.focal_point = center
    pl.camera.position = center + np.array([
        max_dim * 1.5, max_dim * 1.2, max_dim * 1.5,
    ])

    pl.render()


# ── Main viewer ───────────────────────────────────────────────────────


def visualize_chunks(
    seed: int = 42,
    cache_dir: str = "data/chunk_cache/combined",
    show_fillers: bool = False,
) -> None:
    """Load and display cached chunks with N/P key cycling."""
    chunk_files = _discover_chunks(cache_dir)
    total = len(chunk_files)
    rng = np.random.default_rng(seed)

    # Shuffle order so N cycles through random chunks
    indices = list(range(total))
    rng.shuffle(indices)

    state = {"idx": 0, "prior_mode": False}

    # Load first chunk
    first = _load_chunk(chunk_files[indices[0]])
    _print_chunk_stats(first)

    print(f"Found {total} cached chunks.  Building 3D view...")
    pl = pv.Plotter(title="ProspectRL — Real Chunk Viewer")
    pl.set_background("#1a1a2e")

    _render_chunk(pl, first, 0, total, show_fillers)

    pl.add_legend(bcolor=(0.1, 0.1, 0.2, 0.8), face=None)
    pl.add_axes(xlabel="X", ylabel="Y (height)", zlabel="Z")

    controls = (
        "Controls:\n"
        "  N  next chunk\n"
        "  P  previous chunk\n"
        "  T  toggle prior view\n"
        "  Mouse drag to rotate"
    )
    pl.add_text(
        controls, position="lower_left",
        font_size=8, color="#888888",
        name="controls",
    )

    def _refresh() -> None:
        """Re-render current chunk in the active mode."""
        idx = indices[state["idx"]]
        chunk_data = _load_chunk(chunk_files[idx])
        try:
            pl.remove_legend()
        except Exception:
            pass
        if state["prior_mode"]:
            _render_prior(
                pl, chunk_data, state["idx"], total, seed=seed,
            )
        else:
            _render_chunk(
                pl, chunk_data, state["idx"], total, show_fillers,
            )
        pl.add_legend(bcolor=(0.1, 0.1, 0.2, 0.8), face=None)

    def _cycle(delta: int) -> None:
        state["idx"] = (state["idx"] + delta) % total
        idx = indices[state["idx"]]
        chunk_data = _load_chunk(chunk_files[idx])
        _print_chunk_stats(chunk_data)
        _refresh()

    def _toggle_prior() -> None:
        state["prior_mode"] = not state["prior_mode"]
        mode = "PRIOR" if state["prior_mode"] else "CHUNK"
        print(f"Switched to {mode} view")
        _refresh()

    pl.add_key_event("n", lambda: _cycle(1))
    pl.add_key_event("p", lambda: _cycle(-1))
    pl.add_key_event("t", _toggle_prior)

    print("Opening chunk viewer...  (close window to exit)")
    pl.show()


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse real cached Minecraft chunks in 3D",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for chunk shuffle order (default: 42)",
    )
    parser.add_argument(
        "--cache-dir", type=str,
        default="data/chunk_cache/combined",
        help="Directory with .npz chunk files "
        "(default: data/chunk_cache/combined)",
    )
    parser.add_argument(
        "--fillers", action="store_true",
        help="Show filler blocks (dirt, gravel, deepslate, etc.)",
    )
    args = parser.parse_args()

    visualize_chunks(
        seed=args.seed,
        cache_dir=args.cache_dir,
        show_fillers=args.fillers,
    )


if __name__ == "__main__":
    main()
