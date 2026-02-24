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


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class Action(IntEnum):
    FORWARD = 0
    BACK = 1
    UP = 2
    DOWN = 3
    TURN_LEFT = 4
    TURN_RIGHT = 5
    DIG = 6
    DIG_UP = 7
    DIG_DOWN = 8


NUM_ACTIONS: int = len(Action)

# Fuel cost per action (0 = free)
ACTION_FUEL_COST: dict[int, int] = {
    Action.FORWARD: 1,
    Action.BACK: 1,
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


# Keep legacy alias for backward compatibility during transition
OreDistParams = OreSpawnConfig

# -- Ore spawn configs (Minecraft Java Edition parity) ---------------------

ORE_SPAWN_CONFIGS: list[OreSpawnConfig] = [
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
        biomes=[BiomeType.BADLANDS], noise_scale=16, cluster_threshold=0.45,
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
        BlockType.LAPIS_ORE, spawn_size=7, spawn_tries=2,
        y_min_mc=-32, y_max_mc=32, distribution="triangle",
        peak_mc=0, noise_scale=22, cluster_threshold=0.65,
    ),
    OreSpawnConfig(
        BlockType.LAPIS_ORE, spawn_size=7, spawn_tries=4,
        y_min_mc=-64, y_max_mc=64, distribution="uniform",
        air_exposure_skip=1.0, noise_scale=22, cluster_threshold=0.65,
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

# Legacy alias: old code imports ORE_DISTRIBUTIONS — point to new configs
ORE_DISTRIBUTIONS: list[OreSpawnConfig] = ORE_SPAWN_CONFIGS


# ---------------------------------------------------------------------------
# Reward Configuration
# ---------------------------------------------------------------------------

# DEPRECATED — use RewardConfig instead. Kept for backward compatibility.
ORE_BASE_VALUES: dict[int, float] = {
    BlockType.COAL_ORE: 1.0,
    BlockType.IRON_ORE: 2.0,
    BlockType.GOLD_ORE: 4.0,
    BlockType.DIAMOND_ORE: 8.0,
    BlockType.REDSTONE_ORE: 1.5,
    BlockType.EMERALD_ORE: 6.0,
    BlockType.LAPIS_ORE: 1.0,
    BlockType.COPPER_ORE: 1.5,
}

# DEPRECATED — use RewardConfig instead. Kept for backward compatibility.
COST_WEIGHTS: dict[str, float] = {
    "movement": -0.01,
    "dig": -0.005,
    "fuel_penalty": -0.1,     # per step when fuel < 10% max
    "death_penalty": -10.0,   # episode termination from fuel=0
    "time_penalty": -0.001,   # per step, encourages efficiency
    "exploration_bonus": 0.02,  # per step when visiting a new position
}

# DEPRECATED — use RewardConfig instead.
REWARD_ALPHA: float = 1.0


@dataclass
class RewardConfig:
    """Potential-based harvest efficiency reward system.

    Uses an exponential potential function for smooth, stable PPO gradients:
    f(mined, total) = 1 - exp(-mined / (kappa * total + epsilon))
    """

    # Potential-based harvest
    harvest_alpha: float = 8.0         # scale per-step potential delta
    episode_bonus_gamma: float = 3.0   # scale end-of-episode potential bonus
    harvest_kappa: float = 0.4         # exponential saturation parameter
    harvest_epsilon: float = 1.0       # prevents division by zero

    # Adjacent ore penalty (softened with tanh)
    adjacent_penalty_beta: float = 0.05  # base penalty weight
    adjacent_skip_lambda: float = 0.1    # opportunity cost decay per consecutive skip

    # Local clear bonus
    local_clear_bonus: float = 0.2     # bonus when all adjacent desired ores cleared

    # Operational costs
    fuel_penalty: float = -0.1         # per step when fuel < 10% max
    death_penalty: float = -10.0       # episode termination from fuel=0
    time_penalty: float = -0.001       # per step, encourages efficiency


# ---------------------------------------------------------------------------
# Observation Configuration
# ---------------------------------------------------------------------------

# Legacy: used by deployment/inference_server.py (old MLP pipeline)
VOXEL_RADIUS: int = 2
VOXEL_SIZE: int = 2 * VOXEL_RADIUS + 1  # 5
NUM_BLOCK_TYPES: int = len(BlockType)

# ---------------------------------------------------------------------------
# Sliding Window Observation
# ---------------------------------------------------------------------------

# Horizontal extent: 9x9 footprint centered on turtle (radius=4)
OBS_WINDOW_RADIUS_XZ: int = 4
OBS_WINDOW_X: int = 2 * OBS_WINDOW_RADIUS_XZ + 1  # 9
OBS_WINDOW_Z: int = 2 * OBS_WINDOW_RADIUS_XZ + 1  # 9

# Vertical extent: 32 blocks total, biased downward for mining
OBS_WINDOW_Y: int = 32
OBS_WINDOW_Y_ABOVE: int = 8   # blocks above turtle in window
OBS_WINDOW_Y_BELOW: int = 23  # blocks below turtle in window
# Invariant: OBS_WINDOW_Y_ABOVE + 1 + OBS_WINDOW_Y_BELOW == OBS_WINDOW_Y

# Channel groups for voxel encoding
NUM_ORE_CHANNELS: int = NUM_ORE_TYPES  # 8
NUM_VOXEL_CHANNELS: int = NUM_ORE_TYPES + 4 + 1  # ores + solid/soft/air/bedrock + explored = 13

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

# Channel indices
CH_SOLID: int = NUM_ORE_TYPES       # 8
CH_SOFT: int = NUM_ORE_TYPES + 1    # 9
CH_AIR: int = NUM_ORE_TYPES + 2     # 10
CH_BEDROCK: int = NUM_ORE_TYPES + 3  # 11
CH_EXPLORED: int = NUM_ORE_TYPES + 4  # 12

# Scalar observation: pos(3) + facing(4) + fuel(1) + inv(8) + world_h(1)
SCALAR_OBS_DIM: int = 3 + 4 + 1 + NUM_ORE_TYPES + 1  # 17
MAX_WORLD_HEIGHT: int = 384


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


CURRICULUM_STAGES: list[CurriculumStage] = [
    CurriculumStage(
        name="stage1_dense_easy",
        world_size=(20, 40, 20),
        ore_density_multiplier=10.0,
        infinite_fuel=True,
        max_fuel=10000,
        caves_enabled=False,
        preference_mode="one_hot",
        max_episode_steps=500,
        advancement_metric="mean_reward",
        advancement_threshold=25.0,
        advancement_window=100,
    ),
    CurriculumStage(
        name="stage2_sparse_fuel",
        world_size=(32, 64, 32),
        ore_density_multiplier=3.0,
        infinite_fuel=False,
        max_fuel=500,
        caves_enabled=False,
        preference_mode="one_hot",
        max_episode_steps=800,
        advancement_metric="mean_ore_per_episode",
        advancement_threshold=5.0,
        advancement_window=100,
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
]


# ---------------------------------------------------------------------------
# PPO Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """PPO training hyperparameters."""
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training runtime settings."""
    n_envs: int = 4
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
