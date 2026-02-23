"""3D visualization of the generated mining world using PyVista.

Renders ores, caves, and bedrock as colored voxels.  Filler blocks
(dirt, gravel, etc.) are hidden by default but can be shown with
``--fillers``.  Uses curriculum stages from config.py so visualized
worlds match training environments exactly.

Usage::

    python -m prospect_rl.viz_world                    # stage 1 (default)
    python -m prospect_rl.viz_world --stage 2          # stage 2
    python -m prospect_rl.viz_world --stage all        # compare all stages
    python -m prospect_rl.viz_world --seed 123 --stage 3
    python -m prospect_rl.viz_world --stage 1 --biomes --fillers
"""

from __future__ import annotations

import argparse

import numpy as np
import pyvista as pv
from prospect_rl.config import (
    CURRICULUM_STAGES,
    BiomeType,
    BlockType,
    CurriculumStage,
)
from prospect_rl.env.world.world import World

# Minecraft-accurate block colours
BLOCK_COLORS: dict[int, str] = {
    BlockType.COAL_ORE: "#3d3d3d",
    BlockType.IRON_ORE: "#d4a373",
    BlockType.GOLD_ORE: "#fcf051",
    BlockType.DIAMOND_ORE: "#4aedd9",
    BlockType.REDSTONE_ORE: "#ff0000",
    BlockType.EMERALD_ORE: "#17dd62",
    BlockType.LAPIS_ORE: "#345ec3",
    BlockType.COPPER_ORE: "#e07c4a",
    BlockType.BEDROCK: "#555555",
    BlockType.AIR: "#87ceeb",
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
    BlockType.COAL_ORE: "Coal",
    BlockType.IRON_ORE: "Iron",
    BlockType.GOLD_ORE: "Gold",
    BlockType.DIAMOND_ORE: "Diamond",
    BlockType.REDSTONE_ORE: "Redstone",
    BlockType.EMERALD_ORE: "Emerald",
    BlockType.LAPIS_ORE: "Lapis",
    BlockType.COPPER_ORE: "Copper",
    BlockType.BEDROCK: "Bedrock",
    BlockType.AIR: "Air/Cave",
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

# Ore render order (rarest on top for visibility)
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
        geom=pv.Cube(
            x_length=0.9, y_length=0.9, z_length=0.9,
        ),
    )


def _print_stats(
    blocks: np.ndarray, world: World, label: str,
) -> None:
    """Print block distribution statistics to console."""
    sx, sy, sz = blocks.shape
    print(f"\n{'=' * 60}")
    print(f"  {label}: {sx}x{sy}x{sz}")
    print(f"{'=' * 60}")
    total_blocks = blocks.size

    header = (
        f"{'Block Type':<16} {'Count':>8} "
        f"{'% of World':>10}"
    )
    print(f"\n{header}")
    print("-" * 38)

    # Ores
    for bt in ORE_ORDER:
        count = int(np.sum(blocks == bt))
        if count == 0:
            continue
        pct = count / total_blocks * 100
        name = BLOCK_NAMES.get(bt, str(bt))
        print(f"{name:<16} {count:>8,} {pct:>9.3f}%")

    # Fillers
    for bt in FILLER_ORDER:
        count = int(np.sum(blocks == bt))
        if count == 0:
            continue
        pct = count / total_blocks * 100
        name = BLOCK_NAMES.get(bt, str(bt))
        print(f"{name:<16} {count:>8,} {pct:>9.3f}%")

    # Stone, Air, Bedrock
    for bt in [BlockType.STONE, BlockType.AIR, BlockType.BEDROCK]:
        count = int(np.sum(blocks == bt))
        pct = count / total_blocks * 100
        name = BLOCK_NAMES.get(bt, str(bt))
        print(f"{name:<16} {count:>8,} {pct:>9.3f}%")

    # Biome distribution
    bm = world.biome_map
    print(f"\n{'Biome Distribution':}")
    print("-" * 38)
    total_cols = bm.size
    for biome_id in BiomeType:
        count = int(np.sum(bm == biome_id))
        if count == 0:
            continue
        pct = count / total_cols * 100
        name = BIOME_NAMES.get(biome_id, str(biome_id))
        print(f"  {name:<16} {count:>6} cols {pct:>6.1f}%")

    # Y-level ore density profile
    print(f"\n{'Y-Level Ore Density Profile':}")
    print("-" * 40)
    for y in range(0, sy, max(1, sy // 16)):
        y_slice = blocks[:, y, :]
        ore_mask = np.zeros_like(y_slice, dtype=bool)
        for bt in ORE_ORDER:
            ore_mask |= y_slice == bt
        ore_count = int(np.sum(ore_mask))
        total_in_slice = y_slice.size
        bar_len = int(
            ore_count / max(total_in_slice, 1) * 200,
        )
        bar = "#" * bar_len
        print(f"  y={y:>3}: {ore_count:>5} ores  {bar}")
    print()


def _add_world_to_plotter(
    pl: pv.Plotter,
    blocks: np.ndarray,
    show_fillers: bool = False,
) -> None:
    """Add block meshes to a plotter instance."""
    # Render ores (fully opaque)
    for bt in ORE_ORDER:
        mesh = _voxels_for_block(blocks, bt)
        if mesh is not None:
            name = BLOCK_NAMES[bt]
            color = BLOCK_COLORS[bt]
            count = int(np.sum(blocks == bt))
            pl.add_mesh(
                mesh, color=color,
                label=f"{name} ({count:,})",
                opacity=1.0,
            )

    # Render filler blocks (lower opacity)
    if show_fillers:
        for bt in FILLER_ORDER:
            mesh = _voxels_for_block(blocks, bt)
            if mesh is not None:
                name = BLOCK_NAMES[bt]
                color = BLOCK_COLORS[bt]
                count = int(np.sum(blocks == bt))
                pl.add_mesh(
                    mesh, color=color,
                    label=f"{name} ({count:,})",
                    opacity=0.35,
                )

    # Render caves
    air_mesh = _voxels_for_block(blocks, BlockType.AIR)
    if air_mesh is not None:
        air_count = int(np.sum(blocks == BlockType.AIR))
        pl.add_mesh(
            air_mesh,
            color=BLOCK_COLORS[BlockType.AIR],
            label=f"Caves ({air_count:,})",
            opacity=0.15,
        )

    # Render bedrock floor
    bed_mesh = _voxels_for_block(blocks, BlockType.BEDROCK)
    if bed_mesh is not None:
        pl.add_mesh(
            bed_mesh,
            color=BLOCK_COLORS[BlockType.BEDROCK],
            label="Bedrock",
            opacity=0.3,
        )


def _add_biome_overlay(
    pl: pv.Plotter,
    world: World,
) -> None:
    """Add a colored ground-plane showing biome boundaries."""
    bm = world.biome_map
    sx, sz = bm.shape

    for biome_id in BiomeType:
        xs, zs = np.where(bm == biome_id)
        if len(xs) == 0:
            continue
        centers = np.column_stack([
            xs.astype(float),
            np.full(len(xs), -0.5),
            zs.astype(float),
        ])
        cloud = pv.PolyData(centers)
        mesh = cloud.glyph(
            geom=pv.Cube(
                x_length=1.0, y_length=0.1, z_length=1.0,
            ),
        )
        name = BIOME_NAMES.get(biome_id, str(biome_id))
        pl.add_mesh(
            mesh,
            color=BIOME_COLORS[biome_id],
            label=f"Biome: {name}",
            opacity=0.6,
        )


def _stage_label(stage: CurriculumStage) -> str:
    """Build a human-readable label showing stage name and dimensions."""
    sx, sy, sz = stage.world_size
    return (
        f"Stage {CURRICULUM_STAGES.index(stage) + 1}: "
        f"{stage.name}  ({sx}x{sy}x{sz})"
    )


def _world_from_stage(
    stage: CurriculumStage, seed: int,
) -> World:
    """Create a World matching the curriculum stage parameters."""
    return World(
        size=stage.world_size,
        seed=seed,
        ore_density_multiplier=stage.ore_density_multiplier,
        caves_enabled=stage.caves_enabled,
    )


def visualize(
    seed: int = 42,
    stage_idx: int = 0,
    show_all: bool = False,
    show_biomes: bool = False,
    show_fillers: bool = False,
) -> None:
    """Generate world(s) and open a 3D PyVista viewer."""
    if show_all:
        _visualize_all(seed, show_biomes, show_fillers)
        return

    stage = CURRICULUM_STAGES[stage_idx]
    size = stage.world_size
    label = _stage_label(stage)

    print(f"Generating {label} (seed={seed})...")
    world = _world_from_stage(stage, seed)
    blocks = world._blocks
    _print_stats(blocks, world, label)

    print("Building 3D meshes...")
    pl = pv.Plotter(title=f"ProspectRL — {label}")
    pl.set_background("#1a1a2e")

    _add_world_to_plotter(
        pl, blocks, show_fillers=show_fillers,
    )

    if show_biomes:
        _add_biome_overlay(pl, world)

    pl.add_legend(bcolor=(0.1, 0.1, 0.2, 0.8), face=None)
    pl.add_axes(
        xlabel="X", ylabel="Y (height)", zlabel="Z",
    )

    # Dimension overlay in upper-right corner
    sx, sy, sz = size
    dim_text = (
        f"World: {sx}x{sy}x{sz}\n"
        f"Ore density: {stage.ore_density_multiplier}x\n"
        f"Caves: {'ON' if stage.caves_enabled else 'OFF'}\n"
        f"Fuel: {'infinite' if stage.infinite_fuel else stage.max_fuel}\n"
        f"Seed: {seed}"
    )
    pl.add_text(
        dim_text, position="upper_right",
        font_size=10, color="white",
    )
    pl.add_text(
        label, position="upper_left",
        font_size=12, color="white",
    )

    center = np.array([
        size[0] / 2, size[1] / 2, size[2] / 2,
    ])
    max_dim = max(size)
    pl.camera.focal_point = center
    pl.camera.position = center + np.array([
        max_dim * 1.5, max_dim * 1.2, max_dim * 1.5,
    ])

    print("Opening viewer... (close window to exit)")
    pl.show()


def _visualize_all(
    seed: int,
    show_biomes: bool,
    show_fillers: bool = False,
) -> None:
    """Render all curriculum stages side by side."""
    n_stages = len(CURRICULUM_STAGES)

    worlds: list[World] = []
    for stage in CURRICULUM_STAGES:
        label = _stage_label(stage)
        print(f"Generating {label} (seed={seed})...")
        w = _world_from_stage(stage, seed)
        worlds.append(w)
        _print_stats(w._blocks, w, label)

    print("Building 3D meshes for all stages...")
    pl = pv.Plotter(
        shape=(1, n_stages),
        title="ProspectRL — Curriculum Stage Comparison",
    )
    pl.set_background("#1a1a2e")

    for col, stage in enumerate(CURRICULUM_STAGES):
        pl.subplot(0, col)
        w = worlds[col]
        size = stage.world_size
        label = _stage_label(stage)

        _add_world_to_plotter(
            pl, w._blocks, show_fillers=show_fillers,
        )

        if show_biomes:
            _add_biome_overlay(pl, w)

        pl.add_legend(
            bcolor=(0.1, 0.1, 0.2, 0.8), face=None,
        )
        pl.add_axes(
            xlabel="X", ylabel="Y", zlabel="Z",
        )

        sx, sy, sz = size
        info = (
            f"{label}\n"
            f"ore: {stage.ore_density_multiplier}x  "
            f"caves: {'ON' if stage.caves_enabled else 'OFF'}"
        )
        pl.add_text(
            info, position="upper_left",
            font_size=9, color="white",
        )

        center = np.array([
            size[0] / 2, size[1] / 2, size[2] / 2,
        ])
        max_dim = max(size)
        pl.camera.focal_point = center
        pl.camera.position = center + np.array([
            max_dim * 1.5,
            max_dim * 1.2,
            max_dim * 1.5,
        ])

    print("Opening viewer... (close window to exit)")
    pl.show()


def main() -> None:
    # Build stage choices: "1" through "N" plus "all"
    n = len(CURRICULUM_STAGES)
    stage_choices = [str(i + 1) for i in range(n)] + ["all"]
    stage_help_lines = [
        f"  {i + 1}: {s.name} "
        f"({s.world_size[0]}x{s.world_size[1]}x{s.world_size[2]})"
        for i, s in enumerate(CURRICULUM_STAGES)
    ]

    parser = argparse.ArgumentParser(
        description="Visualize mining world (matches curriculum stages)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Curriculum stages:\n" + "\n".join(stage_help_lines),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="World seed (default: 42)",
    )
    parser.add_argument(
        "--stage",
        choices=stage_choices,
        default="1",
        help=(
            f"Curriculum stage 1-{n}, or 'all' to compare "
            f"(default: 1)"
        ),
    )
    parser.add_argument(
        "--biomes", action="store_true",
        help="Show biome overlay on ground plane",
    )
    parser.add_argument(
        "--fillers", action="store_true",
        help="Show filler blocks (dirt, gravel, etc.)",
    )
    args = parser.parse_args()

    show_all = args.stage == "all"
    stage_idx = 0 if show_all else int(args.stage) - 1

    visualize(
        seed=args.seed,
        stage_idx=stage_idx,
        show_all=show_all,
        show_biomes=args.biomes,
        show_fillers=args.fillers,
    )


if __name__ == "__main__":
    main()
