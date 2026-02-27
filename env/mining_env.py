"""Gymnasium environment for ComputerCraft turtle mining with PPO.

Features:
- Dict observation space: ``voxels`` (3D one-hot tensor), ``scalars``
  (global features), and ``pref`` (ore preference)
- Discrete(9) action space with action masking for MaskablePPO
- Sliding 3D observation window (fixed size, independent of world size)
- Stub world for development without Phase 1 world simulation
"""

from __future__ import annotations

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
    CURRICULUM_STAGES,
    MAX_WORLD_HEIGHT,
    NUM_ORE_TYPES,
    NUM_VOXEL_CHANNELS,
    OBS_WINDOW_RADIUS_XZ,
    OBS_WINDOW_X,
    OBS_WINDOW_Y,
    OBS_WINDOW_Y_ABOVE,
    OBS_WINDOW_Y_BELOW,
    OBS_WINDOW_Z,
    ORE_TYPES,
    SCALAR_OBS_DIM,
    SOFT_BLOCKS,
    SOLID_BLOCKS,
    Action,
    BlockType,
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
from prospect_rl.env.turtle import Turtle

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
                    OBS_WINDOW_Y,
                    OBS_WINDOW_X,
                    OBS_WINDOW_Z,
                ),
                dtype=np.float32,
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
        self.action_space = spaces.Discrete(9)

        # Will be initialised in reset()
        self._world: _StubWorld | None = None
        self._turtle: Turtle | None = None
        self._preference: np.ndarray | None = None
        self._explored: set[tuple[int, int, int]] = set()
        self._step_count: int = 0
        self._max_steps: int = self._stage_cfg.max_episode_steps

        # Stage detection
        self._is_stage1: bool = (curriculum_stage == 0)

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
        self._prev_nearest_target_dist: float = float("inf")

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
        if self._world_class is _StubWorld:
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
            )

        # Place turtle at centre of the world (y above bedrock)
        start_pos = np.array(
            [ws[0] // 2, max(1, ws[1] // 2), ws[2] // 2],
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
        self._preference = self._pref_mgr.sample(
            mode=self._preference_mode,
        )

        # Stage 1: resample if target ore doesn't exist in this world
        if self._is_stage1:
            self._preference = self._ensure_achievable_preference(
                self._preference,
            )

        # Explored set
        self._explored = set()
        tp = tuple(int(v) for v in self._turtle.position)
        self._explored.add(tp)

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

        # Stage 1: count target ores, reset waste, compute Y-range
        if self._is_stage1:
            self._total_target_ores_in_world = (
                self._count_target_ores()
            )
            self._cumulative_waste_count = 0
            self._ore_y_range = self._compute_ore_y_range()
            self._prev_nearest_target_dist = float("inf")

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
        if action == Action.FORWARD:
            self._turtle.move_forward(self._world)
        elif action == Action.BACK:
            self._turtle.move_back(self._world)
        elif action == Action.UP:
            self._turtle.move_up(self._world)
        elif action == Action.DOWN:
            self._turtle.move_down(self._world)
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

        # Adjacent weight AFTER action (stages 2-5 only)
        if not self._is_stage1:
            adj_weight_post = (
                self._compute_adjacent_desired_weight()
            )

        # Track explored positions (check newness BEFORE adding)
        tp = tuple(int(v) for v in self._turtle.position)
        is_new_position = tp not in self._explored
        self._explored.add(tp)

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
            curr_dist = self._nearest_target_ore_dist()
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
                turtle_y=int(self._turtle.position[1]),
                ore_y_range=self._ore_y_range,
                world_height=self._stage_cfg.world_size[1],
                prev_nearest_target_dist=(
                    self._prev_nearest_target_dist
                ),
                curr_nearest_target_dist=curr_dist,
            )
            self._prev_nearest_target_dist = curr_dist

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

    def _nearest_target_ore_dist(self) -> float:
        """Manhattan distance to nearest target ore in the obs window.

        Scans the raw world data within the sliding window for blocks
        matching the current preference.  Returns ``float('inf')`` if
        no target ore is visible.
        """
        assert self._turtle is not None and self._world is not None
        assert self._preference is not None

        raw = self._world.get_sliding_window(
            self._turtle.position,
            radius_xz=OBS_WINDOW_RADIUS_XZ,
            y_above=OBS_WINDOW_Y_ABOVE,
            y_below=OBS_WINDOW_Y_BELOW,
        )

        # Build boolean mask of target ore positions in raw (X, Y, Z)
        target_mask = np.zeros(raw.shape, dtype=bool)
        for i, ore_bt in enumerate(ORE_TYPES):
            if self._preference[i] > 0:
                target_mask |= raw == int(ore_bt)

        if not np.any(target_mask):
            return float("inf")

        # Agent is at center of window:
        #   X = OBS_WINDOW_RADIUS_XZ
        #   Y = OBS_WINDOW_Y_BELOW  (raw is X, Y, Z)
        #   Z = OBS_WINDOW_RADIUS_XZ
        center = np.array(
            [OBS_WINDOW_RADIUS_XZ, OBS_WINDOW_Y_BELOW,
             OBS_WINDOW_RADIUS_XZ],
        )
        coords = np.argwhere(target_mask)  # (N, 3)
        dists = np.abs(coords - center).sum(axis=1)
        return float(dists.min())

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
        from prospect_rl.env.turtle import FACING_VECTORS

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
        """Construct the observation dict from current state."""
        assert self._turtle is not None and self._world is not None

        ws = np.array(self._world.shape, dtype=np.float32)
        pos = self._turtle.position.astype(np.float32)

        # --- Voxel tensor (C, Y, X, Z) ---
        voxel_tensor = self._build_voxel_tensor()

        # --- Scalar features ---
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

        scalars = np.concatenate([
            norm_pos, facing_oh, fuel, inv, world_height_norm,
        ])

        return {
            "voxels": voxel_tensor,
            "scalars": scalars,
            "pref": self._preference.copy(),
        }

    def _build_voxel_tensor(self) -> np.ndarray:
        """Build (C, Y, X, Z) multi-channel voxel observation.

        Channels: 0-7 per-ore, 8 solid, 9 soft, 10 air,
        11 bedrock, 12 explored.  Numpy-vectorized.
        """
        assert self._turtle is not None and self._world is not None

        raw = self._world.get_sliding_window(
            self._turtle.position,
            radius_xz=OBS_WINDOW_RADIUS_XZ,
            y_above=OBS_WINDOW_Y_ABOVE,
            y_below=OBS_WINDOW_Y_BELOW,
        )

        # raw: (X, Y, Z) int8 → transpose to (Y, X, Z)
        raw_yxz = raw.transpose(1, 0, 2)

        tensor = np.zeros(
            (
                NUM_VOXEL_CHANNELS,
                OBS_WINDOW_Y,
                OBS_WINDOW_X,
                OBS_WINDOW_Z,
            ),
            dtype=np.float32,
        )

        # Per-ore channels (0 .. NUM_ORE_TYPES-1)
        for i, ore_bt in enumerate(ORE_TYPES):
            tensor[i] = (raw_yxz == int(ore_bt)).astype(
                np.float32,
            )

        # Grouped channels
        tensor[CH_SOLID] = np.isin(
            raw_yxz, _SOLID_ARRAY,
        ).astype(np.float32)
        tensor[CH_SOFT] = np.isin(
            raw_yxz, _SOFT_ARRAY,
        ).astype(np.float32)
        tensor[CH_AIR] = (
            raw_yxz == int(BlockType.AIR)
        ).astype(np.float32)
        tensor[CH_BEDROCK] = (
            raw_yxz == int(BlockType.BEDROCK)
        ).astype(np.float32)

        # Explored mask channel
        tensor[CH_EXPLORED] = self._build_explored_mask()

        # Target ore highlight: fuse per-ore channels weighted by
        # preference so the CNN can detect target ores spatially
        # without needing to learn the pref-to-channel mapping.
        target_mask = np.zeros_like(tensor[0])
        for i in range(NUM_ORE_TYPES):
            if self._preference[i] > 0:
                np.maximum(
                    target_mask,
                    self._preference[i] * tensor[i],
                    out=target_mask,
                )
        tensor[CH_TARGET] = target_mask

        return tensor

    def _build_explored_mask(self) -> np.ndarray:
        """Build explored mask matching (Y, X, Z) window shape."""
        assert self._turtle is not None
        mask = np.zeros(
            (OBS_WINDOW_Y, OBS_WINDOW_X, OBS_WINDOW_Z),
            dtype=np.float32,
        )
        pos = self._turtle.position
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])

        for ex, ey, ez in self._explored:
            wx = ex - px + OBS_WINDOW_RADIUS_XZ
            wy = ey - (py - OBS_WINDOW_Y_BELOW)
            wz = ez - pz + OBS_WINDOW_RADIUS_XZ
            if (
                0 <= wx < OBS_WINDOW_X
                and 0 <= wy < OBS_WINDOW_Y
                and 0 <= wz < OBS_WINDOW_Z
            ):
                mask[wy, wx, wz] = 1.0

        return mask

    def _build_info(self) -> dict[str, Any]:
        """Build the info dict returned alongside observations."""
        assert self._turtle is not None
        return {
            "fuel": self._turtle.fuel,
            "position": self._turtle.position.copy(),
            "facing": self._turtle.facing,
            "step": self._step_count,
            "explored_count": len(self._explored),
        }
