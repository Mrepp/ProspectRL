"""
Centralized configuration for ProspectRL.

All hyperparameters, block definitions, curriculum stages, and reward settings
are defined here. Import via: from prospect_rl.config import Config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

# ---------------------------------------------------------------------------
# Block Registry
# ---------------------------------------------------------------------------

class BlockType(IntEnum):
    """Block IDs used in world simulation. Matches Minecraft block ordering."""
    AIR = 0
    STONE = 1
    DIRT = 2
    BEDROCK = 3
    COAL_ORE = 4
    IRON_ORE = 5
    GOLD_ORE = 6
    DIAMOND_ORE = 7
    REDSTONE_ORE = 8
    EMERALD_ORE = 9
    LAPIS_ORE = 10
    COPPER_ORE = 11
    GRAVEL = 12
    GRANITE = 13
    DIORITE = 14
    ANDESITE = 15
    TUFF = 16
    DEEPSLATE = 17
    CLAY = 18


# Ores that the agent can target (preference vector indices follow this order)
ORE_TYPES: list[BlockType] = [
    BlockType.COAL_ORE,
    BlockType.IRON_ORE,
    BlockType.GOLD_ORE,
    BlockType.DIAMOND_ORE,
    BlockType.REDSTONE_ORE,
    BlockType.EMERALD_ORE,
    BlockType.LAPIS_ORE,
    BlockType.COPPER_ORE,
]

NUM_ORE_TYPES: int = len(ORE_TYPES)

# Mapping from block name strings (as used by Lua client) to BlockType
BLOCK_NAME_TO_ID: dict[str, int] = {
    "air": BlockType.AIR,
    "stone": BlockType.STONE,
    "dirt": BlockType.DIRT,
    "bedrock": BlockType.BEDROCK,
    "coal_ore": BlockType.COAL_ORE,
    "iron_ore": BlockType.IRON_ORE,
    "gold_ore": BlockType.GOLD_ORE,
    "diamond_ore": BlockType.DIAMOND_ORE,
    "redstone_ore": BlockType.REDSTONE_ORE,
    "emerald_ore": BlockType.EMERALD_ORE,
    "lapis_ore": BlockType.LAPIS_ORE,
    "copper_ore": BlockType.COPPER_ORE,
    "gravel": BlockType.GRAVEL,
    "granite": BlockType.GRANITE,
    "diorite": BlockType.DIORITE,
    "andesite": BlockType.ANDESITE,
    "tuff": BlockType.TUFF,
    "deepslate": BlockType.DEEPSLATE,
    "clay": BlockType.CLAY,
}


# ---------------------------------------------------------------------------
# Biome System
# ---------------------------------------------------------------------------

class BiomeType(IntEnum):
    """Simplified biomes for ore distribution differentiation."""
    PLAINS = 0
    MOUNTAINS = 1
    BADLANDS = 2
    DRIPSTONE_CAVES = 3
    LUSH_CAVES = 4


NUM_BIOME_TYPES: int = len(BiomeType)

@dataclass(frozen=True)
class OreTypeConfig:
    """Per-ore-type summary config. One entry per ore in ORE_TYPES order."""
    name: str                                    # e.g. "coal", "diamond"
    block_type: BlockType                        # e.g. BlockType.DIAMOND_ORE
    ore_index: int                               # index in ORE_TYPES (0-7)
    y_min_mc: int                                # union min across spawn configs
    y_max_mc: int                                # union max across spawn configs
    typical_vein_size: int                       # most common spawn_size
    forced_biome: BiomeType | None = None        # None = spawns in all biomes


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class Action(IntEnum):
    FORWARD = 0
    UP = 1
    DOWN = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    DIG = 5
    DIG_UP = 6
    DIG_DOWN = 7


NUM_ACTIONS: int = len(Action)

# Fuel cost per action (0 = free)
ACTION_FUEL_COST: dict[int, int] = {
    Action.FORWARD: 1,
    Action.UP: 1,
    Action.DOWN: 1,
    Action.TURN_LEFT: 0,
    Action.TURN_RIGHT: 0,
    Action.DIG: 0,
    Action.DIG_UP: 0,
    Action.DIG_DOWN: 0,
}


# ---------------------------------------------------------------------------
# Minecraft Y-Coordinate Mapping
# ---------------------------------------------------------------------------

# Minecraft world ranges from y=-64 to y=320 (384 blocks total).
# We map proportionally: sim_y = (mc_y + 64) / 384 * world_height
MC_Y_MIN: int = -64
MC_Y_MAX: int = 320
MC_Y_RANGE: int = MC_Y_MAX - MC_Y_MIN  # 384


# ---------------------------------------------------------------------------
# Ore / Filler Spawn Configuration
# ---------------------------------------------------------------------------

@dataclass
class OreSpawnConfig:
    """A single spawn configuration for an ore or filler block.

    Each block type can have multiple spawn configs with different height
    ranges, distribution types, and biome restrictions — matching Minecraft's
    actual ore generation parameters.
    """
    block_type: BlockType
    spawn_size: int                         # vein size (blocks per attempt)
    spawn_tries: float                      # attempts per chunk (can be fractional)
    y_min_mc: int                           # Minecraft y-coordinate min
    y_max_mc: int                           # Minecraft y-coordinate max
    distribution: str = "triangle"          # "triangle" or "uniform"
    peak_mc: int | None = None              # triangle peak in MC y-coords (None = midpoint)
    air_exposure_skip: float = 0.0          # probability to skip if exposed to air
    biomes: list[BiomeType] | None = None   # None = all biomes
    noise_scale: float = 20.0               # OpenSimplex scale for spatial clustering
    cluster_threshold: float = 0.5          # noise threshold for cluster formation


# -- Ore spawn configs (derived from vanilla 1.21.11 worldgen JSON) --------

# Per-ore clustering parameters for the world generator.  These are
# ProspectRL-specific (no MC JSON equivalent) and preserved for world-gen
# continuity.
_DEFAULT_CLUSTERING: dict[BlockType, tuple[float, float]] = {
    BlockType.COAL_ORE:     (15.0, 0.50),
    BlockType.IRON_ORE:     (18.0, 0.55),
    BlockType.GOLD_ORE:     (20.0, 0.65),
    BlockType.DIAMOND_ORE:  (25.0, 0.70),
    BlockType.REDSTONE_ORE: (18.0, 0.60),
    BlockType.EMERALD_ORE:  (30.0, 0.75),
    BlockType.LAPIS_ORE:    (10.0, 0.45),
    BlockType.COPPER_ORE:   (18.0, 0.55),
}

# Maps MC ore block names → our BlockType enum.
_MC_BLOCK_TO_BLOCKTYPE: dict[str, BlockType] = {
    "coal_ore": BlockType.COAL_ORE,
    "deepslate_coal_ore": BlockType.COAL_ORE,
    "iron_ore": BlockType.IRON_ORE,
    "deepslate_iron_ore": BlockType.IRON_ORE,
    "gold_ore": BlockType.GOLD_ORE,
    "deepslate_gold_ore": BlockType.GOLD_ORE,
    "diamond_ore": BlockType.DIAMOND_ORE,
    "deepslate_diamond_ore": BlockType.DIAMOND_ORE,
    "redstone_ore": BlockType.REDSTONE_ORE,
    "deepslate_redstone_ore": BlockType.REDSTONE_ORE,
    "emerald_ore": BlockType.EMERALD_ORE,
    "deepslate_emerald_ore": BlockType.EMERALD_ORE,
    "lapis_ore": BlockType.LAPIS_ORE,
    "deepslate_lapis_ore": BlockType.LAPIS_ORE,
    "copper_ore": BlockType.COPPER_ORE,
    "deepslate_copper_ore": BlockType.COPPER_ORE,
}


def _worldgen_to_ore_spawn_configs() -> list[OreSpawnConfig]:
    """Generate ORE_SPAWN_CONFIGS from parsed vanilla 1.21.11 worldgen JSON.

    Reads placed/configured feature files under ``data/worldgen/`` and
    converts each relevant placed feature into an :class:`OreSpawnConfig`.
    Falls back to the legacy hardcoded list if the JSON files are not found.
    """
    try:
        from env.worldgen_parser import (
            get_ore_features_per_biome_group,
            load_worldgen,
        )
        wg = load_worldgen()
    except (FileNotFoundError, ImportError):
        # Fall back to empty list — will be caught below
        return _legacy_ore_spawn_configs()

    configs: list[OreSpawnConfig] = []

    # Build biome-group → feature set for biome filtering
    biome_group_features = get_ore_features_per_biome_group(wg.biome_profiles)
    plains_features = biome_group_features.get(0, set())

    for pf in wg.placed_features.values():
        if pf.configured is None:
            continue

        # Map to our BlockType
        block_type = _MC_BLOCK_TO_BLOCKTYPE.get(pf.configured.ore_block)
        if block_type is None:
            continue

        # Determine biome restriction: if this feature is NOT in the
        # plains (default) set, find which biome groups have it.
        biomes: list[BiomeType] | None = None
        if pf.name not in plains_features:
            biome_list: list[BiomeType] = []
            _group_to_biome = {
                1: BiomeType.MOUNTAINS,
                2: BiomeType.BADLANDS,
                3: BiomeType.DRIPSTONE_CAVES,
                4: BiomeType.LUSH_CAVES,
            }
            for gid, biome_enum in _group_to_biome.items():
                if pf.name in biome_group_features.get(gid, set()):
                    biome_list.append(biome_enum)
            if biome_list:
                biomes = biome_list

        # Map distribution type to what OreDistributor expects.
        # Trapezoid with plateau=0 and peak=None (midpoint) is handled
        # by ore_distribution.py's triangle code path.
        dist = pf.height_distribution
        distribution = "triangle" if dist.dist_type == "trapezoid" else "uniform"

        noise_scale, cluster_threshold = _DEFAULT_CLUSTERING.get(
            block_type, (20.0, 0.5),
        )

        configs.append(OreSpawnConfig(
            block_type=block_type,
            spawn_size=pf.configured.size,
            spawn_tries=pf.attempt_model.expected_attempts,
            y_min_mc=dist.min_inclusive,
            y_max_mc=dist.max_inclusive,
            distribution=distribution,
            peak_mc=None,  # Trapezoid is symmetric about midpoint
            air_exposure_skip=pf.configured.discard_chance_on_air_exposure,
            biomes=biomes,
            noise_scale=noise_scale,
            cluster_threshold=cluster_threshold,
        ))

    if not configs:
        return _legacy_ore_spawn_configs()

    return configs


def _legacy_ore_spawn_configs() -> list[OreSpawnConfig]:
    """Hardcoded fallback configs (pre-JSON-parser)."""
    return [
        # Coal (2 configs)
        OreSpawnConfig(
            BlockType.COAL_ORE, spawn_size=17, spawn_tries=20,
            y_min_mc=0, y_max_mc=192, distribution="triangle",
            peak_mc=96, noise_scale=15, cluster_threshold=0.5,
        ),
        OreSpawnConfig(
            BlockType.COAL_ORE, spawn_size=17, spawn_tries=30,
            y_min_mc=136, y_max_mc=256, distribution="uniform",
            noise_scale=12, cluster_threshold=0.45,
        ),
        # Iron (3 configs)
        OreSpawnConfig(
            BlockType.IRON_ORE, spawn_size=9, spawn_tries=10,
            y_min_mc=-64, y_max_mc=72, distribution="uniform",
            noise_scale=18, cluster_threshold=0.55,
        ),
        OreSpawnConfig(
            BlockType.IRON_ORE, spawn_size=9, spawn_tries=10,
            y_min_mc=-24, y_max_mc=56, distribution="triangle",
            peak_mc=16, noise_scale=18, cluster_threshold=0.55,
        ),
        OreSpawnConfig(
            BlockType.IRON_ORE, spawn_size=9, spawn_tries=90,
            y_min_mc=80, y_max_mc=384, distribution="triangle",
            peak_mc=232, noise_scale=14, cluster_threshold=0.4,
        ),
        # Gold (3 configs)
        OreSpawnConfig(
            BlockType.GOLD_ORE, spawn_size=9, spawn_tries=4,
            y_min_mc=-64, y_max_mc=32, distribution="triangle",
            peak_mc=-16, noise_scale=20, cluster_threshold=0.65,
        ),
        OreSpawnConfig(
            BlockType.GOLD_ORE, spawn_size=9, spawn_tries=0.5,
            y_min_mc=-64, y_max_mc=-48, distribution="uniform",
            noise_scale=20, cluster_threshold=0.65,
        ),
        OreSpawnConfig(
            BlockType.GOLD_ORE, spawn_size=9, spawn_tries=50,
            y_min_mc=32, y_max_mc=256, distribution="uniform",
            biomes=[BiomeType.BADLANDS], noise_scale=16,
            cluster_threshold=0.45,
        ),
        # Diamond (3 configs)
        OreSpawnConfig(
            BlockType.DIAMOND_ORE, spawn_size=4, spawn_tries=7,
            y_min_mc=-64, y_max_mc=16, distribution="triangle",
            peak_mc=-59, air_exposure_skip=0.5,
            noise_scale=25, cluster_threshold=0.7,
        ),
        OreSpawnConfig(
            BlockType.DIAMOND_ORE, spawn_size=8, spawn_tries=1,
            y_min_mc=-64, y_max_mc=16, distribution="triangle",
            peak_mc=-59, air_exposure_skip=1.0,
            noise_scale=25, cluster_threshold=0.7,
        ),
        OreSpawnConfig(
            BlockType.DIAMOND_ORE, spawn_size=12, spawn_tries=0.111,
            y_min_mc=-64, y_max_mc=16, distribution="triangle",
            peak_mc=-59, air_exposure_skip=0.7,
            noise_scale=22, cluster_threshold=0.65,
        ),
        # Redstone (2 configs)
        OreSpawnConfig(
            BlockType.REDSTONE_ORE, spawn_size=8, spawn_tries=4,
            y_min_mc=-64, y_max_mc=15, distribution="uniform",
            noise_scale=18, cluster_threshold=0.6,
        ),
        OreSpawnConfig(
            BlockType.REDSTONE_ORE, spawn_size=8, spawn_tries=8,
            y_min_mc=-63, y_max_mc=-32, distribution="triangle",
            peak_mc=-59, noise_scale=18, cluster_threshold=0.6,
        ),
        # Emerald (1 config — Mountains only)
        OreSpawnConfig(
            BlockType.EMERALD_ORE, spawn_size=1, spawn_tries=100,
            y_min_mc=-16, y_max_mc=320, distribution="triangle",
            peak_mc=236, biomes=[BiomeType.MOUNTAINS],
            noise_scale=30, cluster_threshold=0.75,
        ),
        # Lapis Lazuli (2 configs)
        OreSpawnConfig(
            BlockType.LAPIS_ORE, spawn_size=7, spawn_tries=4,
            y_min_mc=-32, y_max_mc=32, distribution="triangle",
            peak_mc=0, noise_scale=10, cluster_threshold=0.45,
        ),
        OreSpawnConfig(
            BlockType.LAPIS_ORE, spawn_size=7, spawn_tries=6,
            y_min_mc=-64, y_max_mc=64, distribution="uniform",
            air_exposure_skip=1.0, noise_scale=10, cluster_threshold=0.45,
        ),
        # Copper (2 configs)
        OreSpawnConfig(
            BlockType.COPPER_ORE, spawn_size=10, spawn_tries=16,
            y_min_mc=-16, y_max_mc=112, distribution="triangle",
            peak_mc=48, noise_scale=18, cluster_threshold=0.55,
        ),
        OreSpawnConfig(
            BlockType.COPPER_ORE, spawn_size=20, spawn_tries=16,
            y_min_mc=-16, y_max_mc=112, distribution="triangle",
            peak_mc=48, biomes=[BiomeType.DRIPSTONE_CAVES],
            noise_scale=16, cluster_threshold=0.5,
        ),
    ]


ORE_SPAWN_CONFIGS: list[OreSpawnConfig] = _worldgen_to_ore_spawn_configs()

# -- Filler block spawn configs -------------------------------------------

FILLER_SPAWN_CONFIGS: list[OreSpawnConfig] = [
    # Dirt
    OreSpawnConfig(
        BlockType.DIRT, spawn_size=33, spawn_tries=7,
        y_min_mc=0, y_max_mc=160, distribution="uniform",
        noise_scale=25, cluster_threshold=0.4,
    ),
    # Clay (Lush Caves only)
    OreSpawnConfig(
        BlockType.CLAY, spawn_size=33, spawn_tries=46,
        y_min_mc=-64, y_max_mc=256, distribution="uniform",
        biomes=[BiomeType.LUSH_CAVES], noise_scale=20, cluster_threshold=0.4,
    ),
    # Gravel
    OreSpawnConfig(
        BlockType.GRAVEL, spawn_size=33, spawn_tries=14,
        y_min_mc=-64, y_max_mc=320, distribution="uniform",
        noise_scale=25, cluster_threshold=0.4,
    ),
    # Granite (2 configs)
    OreSpawnConfig(
        BlockType.GRANITE, spawn_size=64, spawn_tries=2,
        y_min_mc=0, y_max_mc=60, distribution="uniform",
        noise_scale=30, cluster_threshold=0.35,
    ),
    OreSpawnConfig(
        BlockType.GRANITE, spawn_size=64, spawn_tries=0.333,
        y_min_mc=0, y_max_mc=128, distribution="uniform",
        noise_scale=30, cluster_threshold=0.35,
    ),
    # Diorite (2 configs)
    OreSpawnConfig(
        BlockType.DIORITE, spawn_size=64, spawn_tries=2,
        y_min_mc=0, y_max_mc=60, distribution="uniform",
        noise_scale=30, cluster_threshold=0.35,
    ),
    OreSpawnConfig(
        BlockType.DIORITE, spawn_size=64, spawn_tries=0.333,
        y_min_mc=0, y_max_mc=128, distribution="uniform",
        noise_scale=30, cluster_threshold=0.35,
    ),
    # Andesite (2 configs)
    OreSpawnConfig(
        BlockType.ANDESITE, spawn_size=64, spawn_tries=2,
        y_min_mc=0, y_max_mc=60, distribution="uniform",
        noise_scale=30, cluster_threshold=0.35,
    ),
    OreSpawnConfig(
        BlockType.ANDESITE, spawn_size=64, spawn_tries=0.333,
        y_min_mc=0, y_max_mc=128, distribution="uniform",
        noise_scale=30, cluster_threshold=0.35,
    ),
    # Tuff (deepslate layer)
    OreSpawnConfig(
        BlockType.TUFF, spawn_size=64, spawn_tries=2,
        y_min_mc=-64, y_max_mc=0, distribution="uniform",
        noise_scale=30, cluster_threshold=0.35,
    ),
]

def _build_ore_type_configs() -> list[OreTypeConfig]:
    """Derive per-ore-type summary from raw spawn configs."""
    _forced_biomes: dict[BlockType, BiomeType] = {
        BlockType.EMERALD_ORE: BiomeType.MOUNTAINS,
    }
    configs = []
    for i, ore_bt in enumerate(ORE_TYPES):
        matching = [c for c in ORE_SPAWN_CONFIGS if c.block_type == ore_bt]
        y_min = min(c.y_min_mc for c in matching)
        y_max = max(c.y_max_mc for c in matching)
        typical_vein = max(matching, key=lambda c: c.spawn_tries).spawn_size
        name = ore_bt.name.lower().replace("_ore", "")
        configs.append(OreTypeConfig(
            name=name, block_type=ore_bt, ore_index=i,
            y_min_mc=y_min, y_max_mc=y_max,
            typical_vein_size=typical_vein,
            forced_biome=_forced_biomes.get(ore_bt),
        ))
    return configs


ORE_TYPE_CONFIGS: list[OreTypeConfig] = _build_ore_type_configs()


def get_ore_y_ranges(world_height: int) -> list[tuple[float, float]]:
    """Return (y_min, y_max) simulation-Y range per ore type index.

    Derives from ``ORE_TYPE_CONFIGS`` MC Y-ranges, converted to
    simulation coordinates.
    """
    ranges: list[tuple[float, float]] = []
    for otc in ORE_TYPE_CONFIGS:
        sim_min = (otc.y_min_mc - MC_Y_MIN) / MC_Y_RANGE * world_height
        sim_max = (otc.y_max_mc - MC_Y_MIN) / MC_Y_RANGE * world_height
        sim_min = max(0.0, sim_min)
        sim_max = min(float(world_height - 1), sim_max)
        ranges.append((sim_min, sim_max))
    return ranges


# ---------------------------------------------------------------------------
# Reward Configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Potential-based harvest efficiency reward system.

    Uses an exponential potential function for smooth, stable PPO gradients:
    f(mined, total) = 1 - exp(-mined / (kappa * reference_total + epsilon))

    All components are scaled to roughly [-1, +1] per step for stable
    VecNormalize statistics and value function learning.
    """

    # Potential-based harvest
    harvest_alpha: float = 1.0         # scale per-step potential delta
    harvest_kappa: float = 0.4         # exponential saturation parameter
    harvest_epsilon: float = 1.0       # prevents division by zero
    harvest_reference_total: float = 400.0  # fixed ore count for saturation denominator
    potential_maintenance_bonus: float = 0.005  # per-step bonus proportional to potential

    # Adjacent ore penalty (softened with tanh)
    adjacent_penalty_beta: float = 0.5   # base penalty weight
    adjacent_skip_lambda: float = 0.1    # opportunity cost decay per consecutive skip
    adjacent_skip_cap: int = 10          # maximum consecutive skip count

    # Local clear bonus
    local_clear_bonus: float = 0.5     # bonus when all adjacent desired ores cleared

    # Operational costs — progressive fuel curve replaces hard death penalty
    fuel_critical_threshold: float = 0.2  # fuel fraction below which penalty ramps
    fuel_critical_penalty: float = -1.0   # max penalty per step at fuel=0
    time_penalty: float = -0.005       # per step, encourages efficiency

    # Spin penalty — discourages redundant turning (e.g. 3x right instead of 1x left)
    spin_penalty: float = -0.1         # applied when 3+ consecutive same-direction turns


@dataclass
class Stage1RewardConfig:
    """Stage 1 reward: maximize target ore mining.

    Designed for one-shot performance in mining all target ores.
    No fuel constraints. Y-distance penalty guides the agent
    toward the correct depth for its target ore type.
    """

    # Per-target-ore immediate reward
    per_ore_reward: float = 5.0

    # Terminal completion bonus
    completion_scale: float = 10.0

    # Waste penalty (ramps slowly — let agent learn depth-first, refine later)
    waste_beta: float = 0.05
    waste_ramp: int = 200
    waste_alpha: float = 1.5

    # Exploration bonus per new cell visited (small — shouldn't compete
    # with ore targeting)
    exploration_bonus: float = 0.002
    # Fraction of world volume used as half-life: halflife = frac * volume
    exploration_decay_frac: float = 0.003125

    # XZ-plane exploration bonus: rewards visiting new (x, z) columns
    # when the agent is at the correct Y-depth for its target ore.
    xz_exploration_bonus: float = 0.01
    # Fraction of XZ area used as half-life: halflife = frac * xz_area
    xz_exploration_decay_frac: float = 0.2

    # Non-target ore penalty multiplier (wrong ores slightly worse than stone)
    non_target_ore_multiplier: float = 1.5

    # Y-distance penalty: per-step cost when outside target
    # ore's Y range. Quadratic scaling: -scale*(dist/height)^2 - base.
    y_penalty_scale: float = 1.0
    # Constant per-step cost for being off-depth (even by 1 block)
    y_penalty_base: float = 0.05

    # Y-in-range bonus: small per-step reward for being at correct depth
    y_in_range_bonus: float = 0.0

    # Vertical progress shaping: reward for reducing Y-distance to target
    # range. Potential-based: scale * (prev_frac - curr_frac).
    y_progress_scale: float = 1.0

    # One-time bonus when the agent first reaches the target Y-range.
    # Compensates for navigation cost that varies by ore type.
    y_arrival_bonus: float = 3.0

    # Ore discovery bonus: one-time reward when the agent first inspects
    # a cell containing a target ore.
    ore_discovery_bonus: float = 0.5

    # Spin penalty — discourages redundant turning (e.g. 3x right instead of 1x left)
    spin_penalty: float = -0.1

    # No-op penalty — applied when a movement action fails
    noop_penalty: float = -0.05

    # Per-step time penalty — cost to discourage idle looping
    time_penalty: float = -0.01

    # Idle penalty — ramps with steps since last successful dig
    idle_penalty_scale: float = -0.005
    idle_penalty_grace: int = 10

    # Loiter penalty — penalises staying in a small area
    loiter_window: int = 12
    loiter_penalty: float = -0.2
    loiter_unique_threshold: int = 4

    # Dynamic per-ore reward scaling: normalise by world target count
    # so the total available harvest reward is ~constant regardless of
    # ore abundance.  effective_reward = per_ore_reward * (reference / count).
    # Clamped to [harvest_count_floor, harvest_count_ceil] to avoid extremes.
    harvest_reference_count: float = 100.0
    harvest_count_floor: float = 0.25   # minimum multiplier
    harvest_count_ceil: float = 10.0    # maximum multiplier


# ---------------------------------------------------------------------------
# Observation Configuration
# ---------------------------------------------------------------------------

# Legacy: used by deployment/inference_server.py (old MLP pipeline)
VOXEL_RADIUS: int = 2
VOXEL_SIZE: int = 2 * VOXEL_RADIUS + 1  # 5
NUM_BLOCK_TYPES: int = len(BlockType)

# ---------------------------------------------------------------------------
# Fog-of-War Observation
# ---------------------------------------------------------------------------

# Sentinel for unexplored cells in the memory grid
MEMORY_UNKNOWN: int = -1

# Fog-of-war window: smaller than old omniscient window, populated from memory
FOG_WINDOW_RADIUS_XZ: int = 3
FOG_WINDOW_X: int = 2 * FOG_WINDOW_RADIUS_XZ + 1   # 7
FOG_WINDOW_Z: int = 2 * FOG_WINDOW_RADIUS_XZ + 1   # 7
FOG_WINDOW_Y: int = 11
FOG_WINDOW_Y_ABOVE: int = 3    # blocks above turtle in window
FOG_WINDOW_Y_BELOW: int = 7    # blocks below turtle in window
# Invariant: FOG_WINDOW_Y_ABOVE + 1 + FOG_WINDOW_Y_BELOW == FOG_WINDOW_Y

# Channel groups: UNKNOWN + 8 ores + solid/soft/air/bedrock + explored + target = 15
NUM_ORE_CHANNELS: int = NUM_ORE_TYPES  # 8
NUM_VOXEL_CHANNELS: int = 1 + NUM_ORE_TYPES + 4 + 1 + 1  # 15

# Block-to-channel mapping (grouped by dynamics)
SOLID_BLOCKS: frozenset[int] = frozenset({
    int(BlockType.STONE), int(BlockType.DEEPSLATE),
    int(BlockType.GRANITE), int(BlockType.DIORITE),
    int(BlockType.ANDESITE), int(BlockType.TUFF),
})
SOFT_BLOCKS: frozenset[int] = frozenset({
    int(BlockType.DIRT), int(BlockType.GRAVEL),
    int(BlockType.CLAY),
})

# Channel indices (UNKNOWN at 0, ores at 1..8, then grouped)
CH_UNKNOWN: int = 0
# Ore channels: 1 through NUM_ORE_TYPES (1-8)
CH_SOLID: int = 1 + NUM_ORE_TYPES       # 9
CH_SOFT: int = 1 + NUM_ORE_TYPES + 1    # 10
CH_AIR: int = 1 + NUM_ORE_TYPES + 2     # 11
CH_BEDROCK: int = 1 + NUM_ORE_TYPES + 3  # 12
CH_EXPLORED: int = 1 + NUM_ORE_TYPES + 4  # 13
CH_TARGET: int = 1 + NUM_ORE_TYPES + 5   # 14

# Immediate inspection: 3 blocks × NUM_VOXEL_CHANNELS channels each
INSPECT_VECTOR_DIM: int = 3 * NUM_VOXEL_CHANNELS  # 45

# Scalar observation:
#   pos(3) + facing(4) + fuel(1) + inv(8) + world_h(1) + biome(5)  = 22
#   + inspect(45) + fog_density(1) + steps_since_ore(1) + explored_frac(1) = 48
SCALAR_OBS_DIM: int = (
    3 + 4 + 1 + NUM_ORE_TYPES + 1 + NUM_BIOME_TYPES
    + INSPECT_VECTOR_DIM + 3
)  # 70

MAX_WORLD_HEIGHT: int = 384

# ---------------------------------------------------------------------------
# Multi-Agent Extensions
# ---------------------------------------------------------------------------

# Extra voxel channel for agent density in multi-agent mode
CH_AGENT_DENSITY: int = NUM_VOXEL_CHANNELS  # 15
NUM_MULTI_VOXEL_CHANNELS: int = NUM_VOXEL_CHANNELS + 1  # 16

# Multi-agent scalar obs: base 70 + rel_xyz(3) + distance(1) + task_onehot(4)
#   + boundary_dist(1) + inside_flag(1) = 80
SCALAR_OBS_DIM_MULTI: int = SCALAR_OBS_DIM + 10  # 80

# Chunk size for belief map aggregation
CHUNK_SIZE_XZ: int = 16

# Pre-computed ore-type index lookup (block_type_int -> ore_index)
ORE_INDEX: dict[int, int] = {int(bt): i for i, bt in enumerate(ORE_TYPES)}

# Legacy aliases for backward compatibility (deployment/inference_server.py)
OBS_WINDOW_RADIUS_XZ: int = FOG_WINDOW_RADIUS_XZ
OBS_WINDOW_X: int = FOG_WINDOW_X
OBS_WINDOW_Z: int = FOG_WINDOW_Z
OBS_WINDOW_Y: int = FOG_WINDOW_Y
OBS_WINDOW_Y_ABOVE: int = FOG_WINDOW_Y_ABOVE
OBS_WINDOW_Y_BELOW: int = FOG_WINDOW_Y_BELOW


# ---------------------------------------------------------------------------
# Curriculum Stages
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    name: str
    world_size: tuple[int, int, int]
    ore_density_multiplier: float
    infinite_fuel: bool
    max_fuel: int
    caves_enabled: bool
    preference_mode: str  # "one_hot", "two_mix", "dirichlet"
    max_episode_steps: int
    advancement_metric: str
    advancement_threshold: float
    advancement_window: int  # episodes to average over
    # Optional per-ore density overrides: {int(BlockType) -> multiplier}
    ore_density_overrides: dict[int, float] | None = None
    # Coal refueling: fuel restored per coal mined (0 = disabled)
    coal_fuel_value: int = 0
    # Reward bonus for refueling via coal (0.0 = disabled)
    coal_refuel_bonus: float = 0.0


CURRICULUM_STAGES: list[CurriculumStage] = [
    CurriculumStage(
        name="stage1_dense_easy",
        world_size=(40, 40, 40),
        ore_density_multiplier=10.0,
        infinite_fuel=True,
        max_fuel=10000,
        caves_enabled=False,
        preference_mode="one_hot",
        max_episode_steps=1000,
        advancement_metric="mean_reward",
        advancement_threshold=25.0,
        advancement_window=100,
    ),
    CurriculumStage(
        name="stage2_sparse_fuel",
        world_size=(32, 64, 32),
        ore_density_multiplier=3.0,
        infinite_fuel=False,
        max_fuel=200,
        caves_enabled=False,
        preference_mode="one_hot",
        max_episode_steps=2000,
        advancement_metric="mean_ore_per_episode",
        advancement_threshold=5.0,
        advancement_window=100,
        ore_density_overrides={int(BlockType.COAL_ORE): 1.0},
        coal_fuel_value=16,
        coal_refuel_bonus=0.1,
    ),
    CurriculumStage(
        name="stage3_realistic_mixed",
        world_size=(48, 96, 48),
        ore_density_multiplier=1.0,
        infinite_fuel=False,
        max_fuel=500,
        caves_enabled=False,
        preference_mode="two_mix",
        max_episode_steps=1000,
        advancement_metric="mean_reward",
        advancement_threshold=30.0,
        advancement_window=200,
    ),
    CurriculumStage(
        name="stage4_caves_dirichlet",
        world_size=(64, 128, 64),
        ore_density_multiplier=1.0,
        infinite_fuel=False,
        max_fuel=800,
        caves_enabled=True,
        preference_mode="dirichlet",
        max_episode_steps=1500,
        advancement_metric="fuel_efficiency",
        advancement_threshold=0.1,
        advancement_window=200,
    ),
    CurriculumStage(
        name="stage5_full",
        world_size=(128, 256, 128),
        ore_density_multiplier=1.0,
        infinite_fuel=False,
        max_fuel=1000,
        caves_enabled=True,
        preference_mode="dirichlet",
        max_episode_steps=2000,
        advancement_metric="preference_conditioned_eval",
        advancement_threshold=0.7,
        advancement_window=300,
    ),
    CurriculumStage(
        name="stage_real_chunks",
        world_size=(64, 128, 64),
        ore_density_multiplier=1.0,  # Ignored by RealChunkWorld
        infinite_fuel=False,
        max_fuel=800,
        caves_enabled=True,  # Already present in real data
        preference_mode="dirichlet",
        max_episode_steps=1500,
        advancement_metric="mean_reward",
        advancement_threshold=30.0,
        advancement_window=200,
        coal_fuel_value=16,
        coal_refuel_bonus=0.1,
    ),
    CurriculumStage(
        name="stage1_real_eval",
        world_size=(64, 384, 64),       # Full MC Y-range for correct Y-mapping
        ore_density_multiplier=1.0,     # Ignored by RealChunkWorld
        infinite_fuel=True,
        max_fuel=10000,
        caves_enabled=True,             # Real data has caves
        preference_mode="one_hot",
        max_episode_steps=2000,         # Larger world needs more steps
        advancement_metric="mean_reward",
        advancement_threshold=25.0,
        advancement_window=100,
    ),
]


# ---------------------------------------------------------------------------
# PPO Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """PPO training hyperparameters."""
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 2048
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    ent_coef: float = 0.05
    vf_coef: float = 0.75
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    pi_net_arch: list[int] = field(default_factory=lambda: [64, 64])
    vf_net_arch: list[int] = field(default_factory=lambda: [128, 128])


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training runtime settings."""
    n_envs: int = 16
    total_timesteps: int = 1_000_000
    checkpoint_freq: int = 25_000        # save every N steps
    eval_freq: int = 10_000              # evaluate every N steps
    eval_episodes: int = 10
    max_checkpoints_kept: int = 3        # rotate old checkpoints
    tensorboard_log: str = "./tb_logs"
    seed: int = 42


# ---------------------------------------------------------------------------
# Deployment Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeploymentConfig:
    """Inference server settings."""
    host: str = "0.0.0.0"
    port: int = 8080
    model_path: str = "./exports/model.zip"
    vecnormalize_path: str = "./exports/vecnormalize.pkl"


# ---------------------------------------------------------------------------
# Cave Generation Configuration
# ---------------------------------------------------------------------------

@dataclass
class CaveConfig:
    """Parameters for 3D cave generation."""
    noise_scale: float = 30.0
    octaves: int = 2
    threshold: float = 0.4  # higher = fewer caves
    persistence: float = 0.5
    lacunarity: float = 2.0


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Top-level configuration container."""
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    cave: CaveConfig = field(default_factory=CaveConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    stage1_reward: Stage1RewardConfig = field(
        default_factory=Stage1RewardConfig,
    )

    # Derived / convenience
    @property
    def num_ore_types(self) -> int:
        return NUM_ORE_TYPES

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def voxel_shape(self) -> tuple[int, int, int]:
        return (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)

    @property
    def obs_window_shape(self) -> tuple[int, int, int]:
        return (OBS_WINDOW_X, OBS_WINDOW_Y, OBS_WINDOW_Z)

    def get_stage(self, index: int) -> CurriculumStage:
        return CURRICULUM_STAGES[index]

    @property
    def num_stages(self) -> int:
        return len(CURRICULUM_STAGES)
