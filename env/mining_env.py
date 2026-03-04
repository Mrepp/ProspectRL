"""Gymnasium environment for ComputerCraft turtle mining with PPO.

Features:
- Dict observation space: ``voxels`` (3D one-hot tensor), ``scalars``
  (global features), and ``pref`` (ore preference)
- Discrete(8) action space with action masking for MaskablePPO
- Fog-of-war memory: agent inspects 3 blocks/step and builds a map
- Stub world for development without Phase 1 world simulation
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from prospect_rl.config import (
    CH_AIR,
    CH_BEDROCK,
    CH_EXPLORED,
    CH_SOFT,
    CH_SOLID,
    CH_TARGET,
    CH_UNKNOWN,
    CURRICULUM_STAGES,
    FOG_WINDOW_RADIUS_XZ,
    FOG_WINDOW_X,
    FOG_WINDOW_Y,
    FOG_WINDOW_Y_ABOVE,
    FOG_WINDOW_Y_BELOW,
    FOG_WINDOW_Z,
    INSPECT_VECTOR_DIM,
    MAX_WORLD_HEIGHT,
    MEMORY_UNKNOWN,
    NUM_ACTIONS,
    NUM_BIOME_TYPES,
    NUM_ORE_TYPES,
    NUM_VOXEL_CHANNELS,
    ORE_TYPES,
    SCALAR_OBS_DIM,
    SOFT_BLOCKS,
    SOLID_BLOCKS,
    Action,
    BlockType,
    RewardConfig,
    Stage1RewardConfig,
    get_ore_y_ranges,
)
from prospect_rl.env.action_masking import get_action_mask
from prospect_rl.env.preference import PreferenceManager
from prospect_rl.env.reward_vector import (
    _ORE_INDEX,
    compute_reward_components,
    compute_stage1_reward_components,
    compute_stage1_terminal_bonus,
)
from prospect_rl.env.turtle import FACING_VECTORS, Turtle

# ---------------------------------------------------------------------------
# Stub world — simple 3D grid for development before Phase 1 is ready
# ---------------------------------------------------------------------------


class _StubWorld:
    """Minimal 3D block grid filled with stone, random ores, and bedrock."""

    def __init__(
        self,
        size: tuple[int, int, int],
        rng: np.random.Generator,
        ore_density_multiplier: float = 1.0,
    ) -> None:
        sx, sy, sz = size
        self._grid = np.full(
            (sx, sy, sz), BlockType.STONE, dtype=np.int8,
        )
        self._shape = size

        # Bedrock floor at y=0
        self._grid[:, 0, :] = BlockType.BEDROCK

        # Sprinkle ores randomly based on density multiplier
        ore_prob = 0.02 * ore_density_multiplier
        ore_mask = rng.random((sx, sy, sz)) < ore_prob
        ore_mask[:, 0, :] = False
        ore_indices = np.where(ore_mask)
        num_ore_blocks = len(ore_indices[0])
        if num_ore_blocks > 0:
            ore_choices = rng.choice(
                [int(o) for o in ORE_TYPES],
                size=num_ore_blocks,
            )
            self._grid[ore_indices] = ore_choices.astype(np.int8)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def count_blocks(self, block_ids: list[int]) -> int:
        """Count total blocks matching any of the given IDs."""
        mask = np.isin(
            self._grid, np.array(block_ids, dtype=np.int8),
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
        """Extract a sliding window (mirrors World.get_sliding_window)."""
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        sx, sy, sz = self._shape

        x0, x1 = px - radius_xz, px + radius_xz + 1
        y0, y1 = py - y_below, py + y_above + 1
        z0, z1 = pz - radius_xz, pz + radius_xz + 1

        px0 = max(0, -x0)
        px1 = max(0, x1 - sx)
        py0 = max(0, -y0)
        py1 = max(0, y1 - sy)
        pz0 = max(0, -z0)
        pz1 = max(0, z1 - sz)

        chunk = self._grid[
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

        return window

    def __getitem__(self, key: Any) -> Any:
        return self._grid[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._grid[key] = value


# ---------------------------------------------------------------------------
# Try to import the real World class; fall back to stub
# ---------------------------------------------------------------------------

try:
    from prospect_rl.env.world.world import World as _RealWorld
except ImportError:
    _RealWorld = None  # type: ignore[assignment, misc]

try:
    from prospect_rl.env.world.real_chunk_world import RealChunkWorld as _RealChunkWorld
except ImportError:
    _RealChunkWorld = None  # type: ignore[assignment, misc]


# Pre-compute numpy arrays for block-to-channel mapping
_SOLID_ARRAY = np.array(sorted(SOLID_BLOCKS), dtype=np.int8)
_SOFT_ARRAY = np.array(sorted(SOFT_BLOCKS), dtype=np.int8)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MinecraftMiningEnv(gym.Env):
    """Gymnasium env for RL-based turtle mining in a 3D block world.

    Observation space is ``Dict(voxels=Box, scalars=Box, pref=Box)``
    compatible with SB3's ``MaskablePPO`` via a custom feature extractor.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        curriculum_stage: int = 0,
        preference_mode: str | None = None,
        world_class: type | None = None,
        seed: int | None = None,
        fixed_ore_index: int | None = None,
        forced_biome: int | None = None,
        cache_dir: str | None = None,
        required_biome: int | None = None,
    ) -> None:
        super().__init__()

        self._stage_cfg = CURRICULUM_STAGES[curriculum_stage]

        # Allow overriding preference mode (default from curriculum)
        self._preference_mode = (
            preference_mode or self._stage_cfg.preference_mode
        )

        # World class selection
        if world_class is not None:
            self._world_class = world_class
        elif _RealWorld is not None:
            self._world_class = _RealWorld
        else:
            self._world_class = _StubWorld

        # Per-env ore assignment and biome forcing
        self._fixed_ore_index: int | None = fixed_ore_index
        self._forced_biome: int | None = forced_biome

        # Real chunk cache directory (for RealChunkWorld)
        self._cache_dir: str | None = cache_dir
        # Required biome for chunk filtering (for RealChunkWorld)
        self._required_biome: int | None = required_biome

        # RNG
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Preference manager
        self._pref_mgr = PreferenceManager(seed=seed)

        # Spaces
        self.observation_space = spaces.Dict({
            "voxels": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(
                    NUM_VOXEL_CHANNELS,
                    FOG_WINDOW_Y,
                    FOG_WINDOW_X,
                    FOG_WINDOW_Z,
                ),
                dtype=np.float16,
            ),
            "scalars": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(SCALAR_OBS_DIM,),
                dtype=np.float32,
            ),
            "pref": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(NUM_ORE_TYPES,),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Will be initialised in reset()
        self._world: _StubWorld | None = None
        self._turtle: Turtle | None = None
        self._preference: np.ndarray | None = None
        self._explored: set[tuple[int, int, int]] = set()
        self._explored_xz: set[tuple[int, int]] = set()
        self._step_count: int = 0
        self._max_steps: int = self._stage_cfg.max_episode_steps

        # Fog-of-war memory grid (initialised in reset)
        self._memory: np.ndarray | None = None
        # Last inspected block types (front, above, below)
        self._inspected_blocks: tuple[int, int, int] = (
            int(BlockType.STONE), int(BlockType.AIR), int(BlockType.STONE),
        )
        # Discovered target ore positions (for discovery bonus)
        self._discovered_ore_positions: set[tuple[int, int, int]] = set()
        # Steps since agent last inspected a target ore
        self._steps_since_ore_seen: int = 0

        # Stage detection
        self._is_stage1: bool = (
            self._stage_cfg.infinite_fuel
            and self._stage_cfg.preference_mode == "one_hot"
        )

        # Reward state (initialised in reset)
        self._reference_total: float = 0.0
        self._mined_ore_counts = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        self._potential: float = 0.0
        self._prev_adjacent_weight: float = 0.0
        self._consecutive_skip_count: int = 0

        # Stage 1 reward state
        self._cumulative_waste_count: int = 0
        self._total_target_ores_in_world: int = 0
        self._ore_y_range: tuple[float, float] = (0.0, 39.0)
        self._prev_y_dist: float = 0.0

        # Spin detection state
        self._consecutive_turn_count: int = 0
        self._last_turn_direction: int | None = None

        # Loiter detection state (rolling window of recent positions)
        self._recent_positions: deque[tuple[int, int, int]] = deque(
            maxlen=Stage1RewardConfig().loiter_window,
        )

        # Y-arrival bonus state
        self._reached_target_depth: bool = False

        # Idle (non-dig) penalty state
        self._steps_since_last_dig: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._pref_mgr = PreferenceManager(seed=seed)

        ws = self._stage_cfg.world_size

        # Build world
        world_seed = int(self._rng.integers(0, 2**31))
        if (
            _RealChunkWorld is not None
            and self._world_class is _RealChunkWorld
        ):
            self._world = _RealChunkWorld(
                size=ws,
                seed=world_seed,
                cache_dir=self._cache_dir or "data/chunk_cache/default",
                required_biome=self._required_biome,
            )
        elif self._world_class is _StubWorld:
            self._world = _StubWorld(
                size=ws,
                rng=self._rng,
                ore_density_multiplier=(
                    self._stage_cfg.ore_density_multiplier
                ),
            )
        else:
            self._world = self._world_class(
                size=ws,
                seed=world_seed,
                ore_density_multiplier=(
                    self._stage_cfg.ore_density_multiplier
                ),
                caves_enabled=self._stage_cfg.caves_enabled,
                forced_biome=self._forced_biome,
                ore_density_overrides=(
                    self._stage_cfg.ore_density_overrides
                ),
            )

        # Place turtle — use intelligent spawn for RealChunkWorld,
        # randomised Y for procedural worlds
        if (
            _RealChunkWorld is not None
            and isinstance(self._world, _RealChunkWorld)
            and hasattr(self._world, "find_valid_spawn")
        ):
            sx, sy, sz = ws
            spawn_x, spawn_y, spawn_z = self._world.find_valid_spawn(
                sx // 2, sz // 2, rng=self._rng,
            )
            start_pos = np.array(
                [spawn_x, spawn_y, spawn_z], dtype=np.int32,
            )
        else:
            center_y = ws[1] // 2
            jitter_range = max(1, ws[1] // 4)
            spawn_y = int(self._rng.integers(
                max(2, center_y - jitter_range),
                min(ws[1] - 2, center_y + jitter_range) + 1,
            ))
            start_pos = np.array(
                [ws[0] // 2, spawn_y, ws[2] // 2],
                dtype=np.int32,
            )
        # Clear the starting block so the turtle can stand there
        self._world[
            start_pos[0], start_pos[1], start_pos[2]
        ] = BlockType.AIR

        fuel = self._stage_cfg.max_fuel
        self._turtle = Turtle(
            position=start_pos,
            facing=0,
            fuel=fuel,
            max_fuel=self._stage_cfg.max_fuel,
        )

        # Sample preference for this episode
        if self._fixed_ore_index is not None:
            pref = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
            pref[self._fixed_ore_index] = 1.0
            if self._is_stage1:
                pref = self._ensure_achievable_preference(pref)
            self._preference = pref
        else:
            pref = self._pref_mgr.sample(
                mode=self._preference_mode,
            )
            # Stage 1: resample if target ore doesn't exist
            if self._is_stage1:
                pref = self._ensure_achievable_preference(pref)
            self._preference = pref

        # Explored sets
        self._explored = set()
        self._explored_xz = set()
        tp = tuple(int(v) for v in self._turtle.position)
        self._explored.add(tp)
        self._explored_xz.add((tp[0], tp[2]))

        # Fog-of-war memory: all unknown initially
        ws = self._stage_cfg.world_size
        self._memory = np.full(ws, MEMORY_UNKNOWN, dtype=np.int8)
        self._discovered_ore_positions = set()
        self._steps_since_ore_seen = 0

        # Initial inspection: agent position (AIR) + 3 adjacent blocks
        px, py, pz = int(self._turtle.position[0]), int(self._turtle.position[1]), int(self._turtle.position[2])
        self._memory[px, py, pz] = int(BlockType.AIR)
        self._inspected_blocks = self._inspect_three_blocks()

        self._step_count = 0

        # Reward state — reference total from stage config, not per-world
        ws = self._stage_cfg.world_size
        volume = ws[0] * ws[1] * ws[2]
        self._reference_total = (
            volume
            * 0.02
            * self._stage_cfg.ore_density_multiplier
            / NUM_ORE_TYPES
        )
        self._mined_ore_counts = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        self._potential = 0.0
        self._prev_adjacent_weight = 0.0
        self._consecutive_skip_count = 0

        # Spin detection reset
        self._consecutive_turn_count = 0
        self._last_turn_direction = None

        # Loiter detection reset
        self._recent_positions.clear()

        # Idle penalty reset
        self._steps_since_last_dig = 0

        # Arrival bonus reset
        self._reached_target_depth = False

        # Stage 1: count target ores, reset waste, compute Y-range
        if self._is_stage1:
            self._total_target_ores_in_world = (
                self._count_target_ores()
            )
            self._cumulative_waste_count = 0
            self._ore_y_range = self._compute_ore_y_range()
            # Compute initial Y-distance for progress shaping
            spawn_y = int(self._turtle.position[1])
            y_min, y_max = self._ore_y_range
            if spawn_y < y_min:
                self._prev_y_dist = y_min - spawn_y
            elif spawn_y > y_max:
                self._prev_y_dist = spawn_y - y_max
            else:
                self._prev_y_dist = 0.0

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[
        dict[str, np.ndarray], float, bool, bool, dict[str, Any]
    ]:
        assert self._turtle is not None and self._world is not None

        action = int(action)
        block_mined: int | None = None

        # Adjacent weight BEFORE action (stages 2-5 only)
        if not self._is_stage1:
            adj_weight_pre = (
                self._compute_adjacent_desired_weight()
            )

        # Execute action
        move_succeeded = True
        if action == Action.FORWARD:
            move_succeeded = self._turtle.move_forward(self._world)
        elif action == Action.UP:
            move_succeeded = self._turtle.move_up(self._world)
        elif action == Action.DOWN:
            move_succeeded = self._turtle.move_down(self._world)
        elif action == Action.TURN_LEFT:
            self._turtle.turn_left()
        elif action == Action.TURN_RIGHT:
            self._turtle.turn_right()
        elif action == Action.DIG:
            block_mined = self._get_dig_block(action)
            self._turtle.dig(self._world)
        elif action == Action.DIG_UP:
            block_mined = self._get_dig_block(action)
            self._turtle.dig_up(self._world)
        elif action == Action.DIG_DOWN:
            block_mined = self._get_dig_block(action)
            self._turtle.dig_down(self._world)

        # Track consecutive same-direction turns for spin penalty
        if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
            if action == self._last_turn_direction:
                self._consecutive_turn_count += 1
            else:
                self._consecutive_turn_count = 1
                self._last_turn_direction = action
        else:
            self._consecutive_turn_count = 0
            self._last_turn_direction = None

        # --- Fog-of-war memory updates ---
        assert self._memory is not None
        px, py, pz = (
            int(self._turtle.position[0]),
            int(self._turtle.position[1]),
            int(self._turtle.position[2]),
        )
        # Agent's current position is AIR
        self._memory[px, py, pz] = int(BlockType.AIR)
        # After a successful dig, mark dug position as AIR and
        # inspect the block behind (dig-through reveal)
        if block_mined is not None:
            dig_target = self._get_dig_target_pos(action)
            if dig_target is not None:
                dtx, dty, dtz = dig_target
                self._memory[dtx, dty, dtz] = int(BlockType.AIR)
                # Dig-through: inspect block behind the dug block
                behind = self._dig_through_target(action, dig_target)
                if behind is not None:
                    bx, by, bz = behind
                    ws = self._world.shape
                    if (
                        0 <= bx < ws[0]
                        and 0 <= by < ws[1]
                        and 0 <= bz < ws[2]
                    ):
                        self._memory[bx, by, bz] = int(
                            self._world[bx, by, bz],
                        )
        # Inspect 3 blocks (front, above, below) and track ore discovery
        self._inspected_blocks = self._inspect_three_blocks()

        # Adjacent weight AFTER action (stages 2-5 only)
        if not self._is_stage1:
            adj_weight_post = (
                self._compute_adjacent_desired_weight()
            )

        # Track explored positions (check newness BEFORE adding)
        tp = tuple(int(v) for v in self._turtle.position)
        is_new_position = tp not in self._explored
        self._explored.add(tp)

        # Track XZ-plane exploration (new column visits)
        xz = (tp[0], tp[2])
        is_new_xz_position = xz not in self._explored_xz
        self._explored_xz.add(xz)

        # Loiter tracking: record position in rolling window
        self._recent_positions.append(tp)

        # Coal refueling: restore fuel when coal is mined
        self._fuel_restored_this_step = 0
        if (
            block_mined is not None
            and block_mined == int(BlockType.COAL_ORE)
            and self._stage_cfg.coal_fuel_value > 0
            and not self._stage_cfg.infinite_fuel
        ):
            fuel_to_add = self._stage_cfg.coal_fuel_value
            new_fuel = min(
                self._turtle.fuel + fuel_to_add,
                self._turtle.max_fuel,
            )
            self._fuel_restored_this_step = (
                new_fuel - self._turtle.fuel
            )
            self._turtle.fuel = new_fuel

        # Handle infinite fuel
        if self._stage_cfg.infinite_fuel:
            self._turtle.fuel = self._turtle.max_fuel

        # Termination / truncation
        self._step_count += 1
        terminated = (
            self._turtle.fuel == 0
            and not self._stage_cfg.infinite_fuel
        )
        truncated = self._step_count >= self._max_steps

        # Compute reward components (stage-aware)
        if self._is_stage1:
            (
                r_harvest, r_adjacent, r_clear, r_ops,
                self._cumulative_waste_count,
            ) = compute_stage1_reward_components(
                block_mined=block_mined,
                preference=self._preference,
                mined_ore_counts=self._mined_ore_counts,
                cumulative_waste_count=(
                    self._cumulative_waste_count
                ),
                is_new_position=is_new_position,
                explored_count=len(self._explored),
                turtle_y=int(self._turtle.position[1]),
                ore_y_range=self._ore_y_range,
                world_height=self._stage_cfg.world_size[1],
                world_size=self._stage_cfg.world_size,
                total_target_ores_in_world=(
                    self._total_target_ores_in_world
                ),
                is_new_xz_position=is_new_xz_position,
                explored_xz_count=len(self._explored_xz),
                prev_y_dist=self._prev_y_dist,
            )

            # Ore discovery bonus: reward for first-time inspection
            # of a cell containing a target ore
            s1_cfg = Stage1RewardConfig()
            r_clear += self._ore_discovery_reward * s1_cfg.ore_discovery_bonus

            # Update prev_y_dist for next step's progress shaping
            turtle_y_now = int(self._turtle.position[1])
            y_lo, y_hi = self._ore_y_range
            if turtle_y_now < y_lo:
                self._prev_y_dist = y_lo - turtle_y_now
            elif turtle_y_now > y_hi:
                self._prev_y_dist = turtle_y_now - y_hi
            else:
                self._prev_y_dist = 0.0

            # Arrival bonus: one-time reward for reaching target depth
            turtle_y = int(self._turtle.position[1])
            y_min, y_max = self._ore_y_range
            if (
                not self._reached_target_depth
                and y_min <= turtle_y <= y_max
            ):
                self._reached_target_depth = True
                r_clear += s1_cfg.y_arrival_bonus

            # Spin penalty: penalise 3+ consecutive same-direction turns
            if self._consecutive_turn_count >= 3:
                r_ops += s1_cfg.spin_penalty

            # Loiter penalty: penalise staying in a small area
            if len(self._recent_positions) >= s1_cfg.loiter_window:
                unique = len(set(self._recent_positions))
                if unique < s1_cfg.loiter_unique_threshold:
                    r_ops += s1_cfg.loiter_penalty

            # No-op penalty: movement action that failed
            if (
                action in (
                    Action.FORWARD, Action.UP, Action.DOWN,
                )
                and not move_succeeded
            ):
                r_ops += s1_cfg.noop_penalty

            # Idle penalty: ramps with steps since last dig
            if block_mined is not None:
                self._steps_since_last_dig = 0
            else:
                self._steps_since_last_dig += 1
            idle_over = (
                self._steps_since_last_dig - s1_cfg.idle_penalty_grace
            )
            if idle_over > 0:
                r_ops += max(
                    -0.5,
                    s1_cfg.idle_penalty_scale * idle_over,
                )

            # Terminal completion bonus
            terminal_bonus = 0.0
            if terminated or truncated:
                terminal_bonus = compute_stage1_terminal_bonus(
                    mined_ore_counts=self._mined_ore_counts,
                    preference=self._preference,
                    total_target_ores_in_world=(
                        self._total_target_ores_in_world
                    ),
                )

            reward = PreferenceManager.scalarize(
                r_harvest, r_adjacent, r_clear, r_ops,
            ) + terminal_bonus
        else:
            (
                r_harvest, r_adjacent, r_clear, r_ops,
                self._potential,
                self._consecutive_skip_count,
            ) = compute_reward_components(
                action=action,
                block_mined=block_mined,
                turtle=self._turtle,
                max_fuel=self._turtle.max_fuel,
                preference=self._preference,
                reference_total=self._reference_total,
                mined_ore_counts=self._mined_ore_counts,
                prev_potential=self._potential,
                adjacent_desired_weight=adj_weight_pre,
                adjacent_desired_weight_post=adj_weight_post,
                prev_adjacent_weight=(
                    self._prev_adjacent_weight
                ),
                consecutive_skip_count=(
                    self._consecutive_skip_count
                ),
            )
            terminal_bonus = 0.0

            # Spin penalty: penalise 3+ consecutive same-direction turns
            if self._consecutive_turn_count >= 3:
                spin_cfg = RewardConfig()
                r_ops += spin_cfg.spin_penalty

            # Coal refuel bonus: incentivize mining coal for fuel
            if (
                self._fuel_restored_this_step > 0
                and self._stage_cfg.coal_refuel_bonus > 0
            ):
                fuel_frac = self._turtle.fuel / max(
                    self._turtle.max_fuel, 1,
                )
                need_factor = 1.0 - fuel_frac
                r_ops += self._stage_cfg.coal_refuel_bonus * (
                    0.5 + 0.5 * need_factor
                )

            reward = PreferenceManager.scalarize(
                r_harvest, r_adjacent, r_clear, r_ops,
            )

            # Update state for next step (stages 2-5)
            self._prev_adjacent_weight = adj_weight_post

        obs = self._build_obs()
        info = self._build_info()
        info["r_harvest"] = r_harvest
        info["r_adjacent"] = r_adjacent
        info["r_clear"] = r_clear
        info["r_ops"] = r_ops
        info["harvest_potential"] = self._potential
        info["block_mined"] = block_mined
        info["is_stage1"] = self._is_stage1
        info["fuel_restored"] = self._fuel_restored_this_step

        # Stage 1 specific info
        if self._is_stage1:
            target_mined = int(np.dot(
                self._preference > 0,
                self._mined_ore_counts,
            ))
            info["target_ores_mined"] = target_mined
            info["total_target_in_world"] = (
                self._total_target_ores_in_world
            )
            info["completion_ratio"] = (
                target_mined
                / max(1, self._total_target_ores_in_world)
            )
            info["cumulative_waste"] = (
                self._cumulative_waste_count
            )
            info["terminal_bonus"] = terminal_bonus

        return obs, float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions (MaskablePPO)."""
        assert self._turtle is not None and self._world is not None
        return get_action_mask(
            self._turtle,
            self._world,
            self._world.shape,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_achievable_preference(
        self,
        preference: np.ndarray,
        max_attempts: int = 20,
    ) -> np.ndarray:
        """Resample preference if the target ore has 0 blocks in the world.

        Falls back to the most common ore type after max_attempts.
        """
        assert self._world is not None
        for _ in range(max_attempts):
            target_ids = [
                int(ORE_TYPES[i])
                for i in range(len(ORE_TYPES))
                if preference[i] > 0
            ]
            if target_ids and self._world.count_blocks(target_ids) > 0:
                return preference
            preference = self._pref_mgr.sample(
                mode=self._preference_mode,
            )

        # Fallback: find the most common ore and target it
        best_idx = 0
        best_count = 0
        for i, ore_bt in enumerate(ORE_TYPES):
            count = self._world.count_blocks([int(ore_bt)])
            if count > best_count:
                best_count = count
                best_idx = i
        fallback = np.zeros(len(ORE_TYPES), dtype=np.float32)
        fallback[best_idx] = 1.0
        return fallback

    def _compute_ore_y_range(self) -> tuple[float, float]:
        """Compute the target ore's Y-range for the current preference."""
        assert self._preference is not None
        world_h = self._stage_cfg.world_size[1]
        all_ranges = get_ore_y_ranges(world_h)
        # For one-hot: single target ore index
        target_idx = int(np.argmax(self._preference))
        return all_ranges[target_idx]

    def _count_target_ores(self) -> int:
        """Count target ores in the world for the current preference."""
        assert self._world is not None
        assert self._preference is not None
        target_ids = [
            int(ORE_TYPES[i])
            for i in range(len(ORE_TYPES))
            if self._preference[i] > 0
        ]
        if not target_ids:
            return 0
        return self._world.count_blocks(target_ids)

    # ------------------------------------------------------------------
    # Fog-of-war helpers
    # ------------------------------------------------------------------

    def _inspect_three_blocks(self) -> tuple[int, int, int]:
        """Inspect front/above/below blocks and update memory.

        Returns the block types at (front, above, below). Also tracks
        new target ore discoveries and updates ``_ore_discovery_reward``.
        """
        assert self._turtle is not None and self._world is not None
        assert self._memory is not None
        assert self._preference is not None

        pos = self._turtle.position
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        ws = self._world.shape
        self._ore_discovery_reward = 0

        def _inspect(x: int, y: int, z: int) -> int:
            if 0 <= x < ws[0] and 0 <= y < ws[1] and 0 <= z < ws[2]:
                block = int(self._world[x, y, z])
                self._memory[x, y, z] = block
                # Track target ore discovery
                if block in _ORE_INDEX:
                    idx = _ORE_INDEX[block]
                    if self._preference[idx] > 0:
                        self._steps_since_ore_seen = 0
                        pos_t = (x, y, z)
                        if pos_t not in self._discovered_ore_positions:
                            self._discovered_ore_positions.add(pos_t)
                            self._ore_discovery_reward += 1
                return block
            return int(BlockType.BEDROCK)

        # Front block
        fv = FACING_VECTORS[self._turtle.facing]
        front = _inspect(px + fv[0], py + fv[1], pz + fv[2])
        # Above block
        above = _inspect(px, py + 1, pz)
        # Below block
        below = _inspect(px, py - 1, pz)

        self._steps_since_ore_seen += 1
        return front, above, below

    def _get_dig_target_pos(
        self, action: int,
    ) -> tuple[int, int, int] | None:
        """Return the world position targeted by a dig action."""
        assert self._turtle is not None
        pos = self._turtle.position
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        if action == Action.DIG:
            fv = FACING_VECTORS[self._turtle.facing]
            return (px + fv[0], py + fv[1], pz + fv[2])
        if action == Action.DIG_UP:
            return (px, py + 1, pz)
        if action == Action.DIG_DOWN:
            return (px, py - 1, pz)
        return None

    def _dig_through_target(
        self,
        action: int,
        dig_pos: tuple[int, int, int],
    ) -> tuple[int, int, int] | None:
        """Return the position behind a dug block for dig-through reveal."""
        dx, dy, dz = dig_pos
        if action == Action.DIG:
            fv = FACING_VECTORS[self._turtle.facing]
            return (dx + fv[0], dy + fv[1], dz + fv[2])
        if action == Action.DIG_UP:
            return (dx, dy + 1, dz)
        if action == Action.DIG_DOWN:
            return (dx, dy - 1, dz)
        return None

    def _get_memory_window(self) -> np.ndarray:
        """Extract a fog-of-war observation window from memory.

        Returns shape ``(FOG_WINDOW_X, FOG_WINDOW_Y, FOG_WINDOW_Z)``
        as int8. Unexplored cells have value ``MEMORY_UNKNOWN``.
        Out-of-bounds cells are also ``MEMORY_UNKNOWN``.
        """
        assert self._turtle is not None and self._memory is not None
        pos = self._turtle.position
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        sx, sy, sz = self._memory.shape

        r = FOG_WINDOW_RADIUS_XZ
        x0, x1 = px - r, px + r + 1
        y0, y1 = py - FOG_WINDOW_Y_BELOW, py + FOG_WINDOW_Y_ABOVE + 1
        z0, z1 = pz - r, pz + r + 1

        # Compute padding for out-of-bounds
        px0 = max(0, -x0)
        px1 = max(0, x1 - sx)
        py0 = max(0, -y0)
        py1 = max(0, y1 - sy)
        pz0 = max(0, -z0)
        pz1 = max(0, z1 - sz)

        chunk = self._memory[
            max(0, x0):min(sx, x1),
            max(0, y0):min(sy, y1),
            max(0, z0):min(sz, z1),
        ]

        if px0 or px1 or py0 or py1 or pz0 or pz1:
            window = np.pad(
                chunk,
                ((px0, px1), (py0, py1), (pz0, pz1)),
                mode="constant",
                constant_values=MEMORY_UNKNOWN,
            )
        else:
            window = chunk.copy()

        return window

    @staticmethod
    def _encode_block_one_hot(block_type: int) -> np.ndarray:
        """Encode a single block as a NUM_VOXEL_CHANNELS one-hot vector."""
        vec = np.zeros(NUM_VOXEL_CHANNELS, dtype=np.float32)
        if block_type == MEMORY_UNKNOWN:
            vec[CH_UNKNOWN] = 1.0
        elif block_type in _ORE_INDEX:
            vec[1 + _ORE_INDEX[block_type]] = 1.0
        elif block_type == int(BlockType.AIR):
            vec[CH_AIR] = 1.0
        elif block_type == int(BlockType.BEDROCK):
            vec[CH_BEDROCK] = 1.0
        elif block_type in SOLID_BLOCKS:
            vec[CH_SOLID] = 1.0
        elif block_type in SOFT_BLOCKS:
            vec[CH_SOFT] = 1.0
        else:
            # Unknown block type → treat as solid
            vec[CH_SOLID] = 1.0
        return vec

    def _compute_adjacent_desired_weight(self) -> float:
        """Sum preference weights of desired ores adjacent to the turtle.

        Checks 6 cardinal neighbors: +-x, +-y, +-z.
        """
        assert self._turtle is not None and self._world is not None
        pos = self._turtle.position
        ws = self._world.shape

        offsets = (
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        )

        total_weight = 0.0
        for dx, dy, dz in offsets:
            nx, ny, nz = int(pos[0]) + dx, int(pos[1]) + dy, int(pos[2]) + dz
            if nx < 0 or ny < 0 or nz < 0:
                continue
            if nx >= ws[0] or ny >= ws[1] or nz >= ws[2]:
                continue
            block = int(self._world[nx, ny, nz])
            if block in _ORE_INDEX:
                idx = _ORE_INDEX[block]
                if self._preference[idx] > 0:
                    total_weight += float(self._preference[idx])

        return total_weight

    def _get_dig_block(self, action: int) -> int | None:
        """Return block type at the dig target, or None."""
        assert self._turtle is not None and self._world is not None
        pos = self._turtle.position

        if action == Action.DIG:
            offset = FACING_VECTORS[self._turtle.facing]
        elif action == Action.DIG_UP:
            offset = np.array([0, 1, 0], dtype=np.int32)
        else:  # DIG_DOWN
            offset = np.array([0, -1, 0], dtype=np.int32)

        target = pos + offset
        ws = np.array(self._world.shape, dtype=np.int32)
        if np.any(target < 0) or np.any(target >= ws):
            return None
        block = int(
            self._world[target[0], target[1], target[2]],
        )
        if block == BlockType.AIR or block == BlockType.BEDROCK:
            return None
        return block

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Construct the observation dict from current state.

        Scalars layout (70 dims):
          [0:3]   normalized position (x, y, z)
          [3:7]   facing one-hot (4)
          [7:8]   fuel fraction (1)
          [8:16]  ore inventory (8)
          [16:17] world height normalized (1)
          [17:22] biome one-hot (5)
          [22:37] front block inspection (15-ch one-hot)
          [37:52] above block inspection (15-ch one-hot)
          [52:67] below block inspection (15-ch one-hot)
          [67]    fog density (fraction of memory window that is unknown)
          [68]    steps since last ore seen (normalized)
          [69]    total explored fraction
        """
        assert self._turtle is not None and self._world is not None

        ws = np.array(self._world.shape, dtype=np.float32)
        pos = self._turtle.position.astype(np.float32)

        # --- Voxel tensor (C, Y, X, Z) ---
        voxel_tensor = self._build_voxel_tensor()

        # --- Scalar features (first 22: same as before) ---
        norm_pos = pos / ws

        facing_oh = np.zeros(4, dtype=np.float32)
        facing_oh[self._turtle.facing] = 1.0

        fuel = np.array(
            [self._turtle.fuel / max(self._turtle.max_fuel, 1)],
            dtype=np.float32,
        )

        inv = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        for i, ore_bt in enumerate(ORE_TYPES):
            inv[i] = float(
                self._turtle.inventory.get(int(ore_bt), 0),
            )

        world_height_norm = np.array(
            [ws[1] / MAX_WORLD_HEIGHT], dtype=np.float32,
        )

        # Biome at agent's XZ position (one-hot)
        biome_oh = np.zeros(NUM_BIOME_TYPES, dtype=np.float32)
        bx = min(int(pos[0]), self._world.biome_map.shape[0] - 1)
        bz = min(int(pos[2]), self._world.biome_map.shape[1] - 1)
        biome_id = int(self._world.biome_map[bx, bz])
        if 0 <= biome_id < NUM_BIOME_TYPES:
            biome_oh[biome_id] = 1.0

        # --- Inspection vectors (45 dims: 3 blocks × 15 channels) ---
        front_bt, above_bt, below_bt = self._inspected_blocks
        inspect_front = self._encode_block_one_hot(front_bt)
        inspect_above = self._encode_block_one_hot(above_bt)
        inspect_below = self._encode_block_one_hot(below_bt)

        # --- Fog density: fraction of memory window that is unknown ---
        mem_window = self._get_memory_window()
        total_cells = mem_window.size
        unknown_cells = int(np.sum(mem_window == MEMORY_UNKNOWN))
        fog_density = np.array(
            [unknown_cells / max(total_cells, 1)],
            dtype=np.float32,
        )

        # --- Steps since last ore seen (normalized by max_steps) ---
        max_steps = self._stage_cfg.max_episode_steps
        steps_since_ore = np.array(
            [min(self._steps_since_ore_seen, max_steps)
             / max(max_steps, 1)],
            dtype=np.float32,
        )

        # --- Total explored fraction ---
        world_volume = ws[0] * ws[1] * ws[2]
        explored_frac = np.array(
            [len(self._explored) / max(world_volume, 1)],
            dtype=np.float32,
        )

        scalars = np.concatenate([
            norm_pos, facing_oh, fuel, inv, world_height_norm,
            biome_oh,
            inspect_front, inspect_above, inspect_below,
            fog_density, steps_since_ore, explored_frac,
        ])

        return {
            "voxels": voxel_tensor,
            "scalars": scalars,
            "pref": self._preference.copy(),
        }

    def _build_voxel_tensor(self) -> np.ndarray:
        """Build (C, Y, X, Z) multi-channel voxel observation from memory.

        Channels: 0 unknown, 1-8 per-ore, 9 solid, 10 soft, 11 air,
        12 bedrock, 13 explored, 14 target.  Numpy-vectorized.
        """
        assert self._turtle is not None and self._memory is not None

        # raw: (X, Y, Z) int8 from fog-of-war memory
        raw = self._get_memory_window()
        # Transpose to (Y, X, Z) for channel-first layout
        raw_yxz = raw.transpose(1, 0, 2)

        tensor = np.zeros(
            (
                NUM_VOXEL_CHANNELS,
                FOG_WINDOW_Y,
                FOG_WINDOW_X,
                FOG_WINDOW_Z,
            ),
            dtype=np.float16,
        )

        # Channel 0: UNKNOWN (fog)
        tensor[CH_UNKNOWN] = (
            raw_yxz == MEMORY_UNKNOWN
        ).astype(np.float16)

        # Per-ore channels (1 .. NUM_ORE_TYPES)
        for i, ore_bt in enumerate(ORE_TYPES):
            tensor[1 + i] = (raw_yxz == int(ore_bt)).astype(
                np.float16,
            )

        # Grouped channels
        tensor[CH_SOLID] = np.isin(
            raw_yxz, _SOLID_ARRAY,
        ).astype(np.float16)
        tensor[CH_SOFT] = np.isin(
            raw_yxz, _SOFT_ARRAY,
        ).astype(np.float16)
        tensor[CH_AIR] = (
            raw_yxz == int(BlockType.AIR)
        ).astype(np.float16)
        tensor[CH_BEDROCK] = (
            raw_yxz == int(BlockType.BEDROCK)
        ).astype(np.float16)

        # Explored mask channel (physically visited positions)
        tensor[CH_EXPLORED] = self._build_explored_mask()

        # Target ore highlight: fuse per-ore channels weighted by
        # preference so the CNN can detect target ores spatially
        # without needing to learn the pref-to-channel mapping.
        target_mask = np.zeros_like(tensor[0])
        for i in range(NUM_ORE_TYPES):
            if self._preference[i] > 0:
                np.maximum(
                    target_mask,
                    self._preference[i] * tensor[1 + i],
                    out=target_mask,
                )
        tensor[CH_TARGET] = target_mask

        return tensor

    def _build_explored_mask(self) -> np.ndarray:
        """Build explored mask matching (Y, X, Z) window shape.

        Uses vectorized numpy operations instead of a Python loop.
        """
        assert self._turtle is not None
        if not self._explored:
            return np.zeros(
                (FOG_WINDOW_Y, FOG_WINDOW_X, FOG_WINDOW_Z),
                dtype=np.float16,
            )

        pos = self._turtle.position
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])

        coords = np.array(list(self._explored), dtype=np.int32)
        wx = coords[:, 0] - px + FOG_WINDOW_RADIUS_XZ
        wy = coords[:, 1] - (py - FOG_WINDOW_Y_BELOW)
        wz = coords[:, 2] - pz + FOG_WINDOW_RADIUS_XZ

        valid = (
            (wx >= 0) & (wx < FOG_WINDOW_X)
            & (wy >= 0) & (wy < FOG_WINDOW_Y)
            & (wz >= 0) & (wz < FOG_WINDOW_Z)
        )

        mask = np.zeros(
            (FOG_WINDOW_Y, FOG_WINDOW_X, FOG_WINDOW_Z),
            dtype=np.float16,
        )
        mask[wy[valid], wx[valid], wz[valid]] = 1.0
        return mask

    def _build_info(self) -> dict[str, Any]:
        """Build the info dict returned alongside observations."""
        assert self._turtle is not None
        world_type = (
            "real"
            if _RealChunkWorld is not None
            and isinstance(self._world, _RealChunkWorld)
            else "sim"
        )
        return {
            "fuel": self._turtle.fuel,
            "position": self._turtle.position.copy(),
            "facing": self._turtle.facing,
            "step": self._step_count,
            "explored_count": len(self._explored),
            "explored_xz_count": len(self._explored_xz),
            "world_type": world_type,
        }
