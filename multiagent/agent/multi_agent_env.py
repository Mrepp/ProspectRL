"""Per-agent environment for multi-agent mining.

Each ``MultiAgentMiningEnv`` instance represents one agent operating
in a ``SharedWorld``. Multiple instances share the same world and
coordinate via the belief map and coordinator.

Features:
- Hybrid A*/RL navigation (A* when far, RL when near target)
- Telemetry event emission on every step
- Observation mismatch detection (BLOCK_CHANGED, BLOCK_ADDED)
- Bounding box observation and reward shaping
- Task-specific reward computation
"""

from __future__ import annotations

import math
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
    ORE_INDEX as _ORE_INDEX,
    ORE_TYPES,
    SCALAR_OBS_DIM,
    SCALAR_OBS_DIM_MULTI,
    SOFT_BLOCKS,
    SOLID_BLOCKS,
    Action,
    BlockType,
)
from prospect_rl.env.action_masking import get_action_mask
from prospect_rl.env.turtle import FACING_VECTORS, Turtle
from prospect_rl.multiagent.agent.communication import (
    AgentMessage,
    MessageBuffer,
    MessageType,
)
from prospect_rl.multiagent.agent.task_rewards import compute_task_reward
from prospect_rl.multiagent.coordinator.assignment import (
    NUM_TASK_TYPES,
    BoundingBox,
    TaskAssignment,
    TaskType,
)
from prospect_rl.multiagent.pathfinding import AStarPathfinder
from prospect_rl.multiagent.shared_world import SharedWorld
from prospect_rl.multiagent.telemetry import TelemetryEvent, TelemetryEventType

# Multi-agent voxel channels: 15 original + 1 agent density = 16
NUM_MULTI_VOXEL_CHANNELS = NUM_VOXEL_CHANNELS + 1
CH_AGENT_DENSITY = NUM_VOXEL_CHANNELS  # Channel 15

# Pre-compute arrays for block-to-channel mapping
_SOLID_ARRAY = np.array(sorted(SOLID_BLOCKS), dtype=np.int8)
_SOFT_ARRAY = np.array(sorted(SOFT_BLOCKS), dtype=np.int8)


class MultiAgentMiningEnv(gym.Env):
    """Per-agent environment for multi-agent mining.

    Parameters
    ----------
    agent_id:
        Unique identifier for this agent.
    shared_world:
        The shared world instance.
    mining_radius:
        Distance threshold for A*/RL mode switch.
    max_episode_steps:
        Maximum steps per episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_id: int,
        shared_world: SharedWorld,
        mining_radius: int = 8,
        max_episode_steps: int = 1000,
        task_reward_config: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        self._agent_id = agent_id
        self._shared_world = shared_world
        self._mining_radius = mining_radius
        self._max_steps = max_episode_steps
        self._task_reward_config = task_reward_config or {}

        # Observation space (multi-agent extended)
        self.observation_space = spaces.Dict({
            "voxels": spaces.Box(
                low=0.0, high=1.0,
                shape=(
                    NUM_MULTI_VOXEL_CHANNELS,
                    FOG_WINDOW_Y,
                    FOG_WINDOW_X,
                    FOG_WINDOW_Z,
                ),
                dtype=np.float16,
            ),
            "scalars": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(SCALAR_OBS_DIM_MULTI,),
                dtype=np.float32,
            ),
            "pref": spaces.Box(
                low=0.0, high=1.0,
                shape=(NUM_ORE_TYPES,),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Agent state (set in reset)
        self._turtle: Turtle | None = None
        self._assignment: TaskAssignment | None = None
        self._preference = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        self._step_count = 0
        self._task_steps = 0
        self._blocks_cleared = 0

        # Fog-of-war memory
        self._memory: np.ndarray | None = None
        self._explored: set[tuple[int, int, int]] = set()
        self._inspected_blocks: tuple[int, int, int] = (
            int(BlockType.STONE), int(BlockType.AIR), int(BlockType.STONE),
        )

        # Pathfinder
        self._pathfinder = AStarPathfinder()

        # Communication
        self._message_buffer = MessageBuffer()

        # Telemetry events for this step
        self._step_events: list[TelemetryEvent] = []

        # Steps since agent last mined an ore block
        self._steps_since_ore: int = 0

        # Distribution tracking for MINE_ORE tasks
        self._mined_ore_counts = np.zeros(
            NUM_ORE_TYPES, dtype=np.int32,
        )
        self._prev_alignment: float = 0.0

        # Previous position for reward computation
        self._prev_position: np.ndarray | None = None

        # World diagonal for normalization
        sx, sy, sz = shared_world.shape
        self._world_diagonal = math.sqrt(sx * sx + sy * sy + sz * sz)

    # ------------------------------------------------------------------
    # Task assignment
    # ------------------------------------------------------------------

    def set_assignment(self, assignment: TaskAssignment) -> None:
        """Set a new task assignment from the coordinator."""
        self._assignment = assignment
        self._preference = assignment.ore_preference.copy()
        self._task_steps = 0
        self._blocks_cleared = 0
        self._mined_ore_counts[:] = 0
        self._prev_alignment = 0.0

        # Compute A* path for navigation tasks
        if assignment.is_navigation_task and self._turtle is not None:
            path = self._pathfinder.find_path(
                self._turtle.position,
                assignment.target_position,
                self._shared_world,
                self._agent_id,
            )
            if path is not None:
                self._pathfinder.set_path(path, assignment.target_position)
            else:
                self._pathfinder.invalidate_path()

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

        self._step_count = 0
        self._task_steps = 0
        self._blocks_cleared = 0
        self._steps_since_ore = 0
        self._mined_ore_counts[:] = 0
        self._prev_alignment = 0.0
        self._explored.clear()
        self._step_events.clear()
        self._message_buffer.clear()
        self._pathfinder.invalidate_path()

        # Initialize memory grid
        ws = self._shared_world.shape
        self._memory = np.full(ws, MEMORY_UNKNOWN, dtype=np.int8)

        # Turtle should already be registered in shared_world
        turtle = self._shared_world.get_agent(self._agent_id)
        if turtle is not None:
            self._turtle = turtle
        else:
            # Create a default turtle if not registered
            pos = np.array([ws[0] // 2, ws[1] // 2, ws[2] // 2], dtype=np.int32)
            self._shared_world[pos[0], pos[1], pos[2]] = BlockType.AIR
            self._turtle = Turtle(position=pos)
            self._shared_world.register_agent(self._agent_id, self._turtle)

        # Mark starting position
        tp = tuple(int(v) for v in self._turtle.position)
        self._explored.add(tp)
        px, py, pz = tp
        self._memory[px, py, pz] = int(BlockType.AIR)

        # Initial inspection
        self._inspected_blocks = self._inspect_three_blocks()
        self._prev_position = self._turtle.position.copy()

        obs = self._build_obs()
        return obs, self._build_info()

    def step(
        self, action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self._turtle is not None and self._memory is not None

        action = int(action)
        self._step_events.clear()
        block_mined: int | None = None

        # Save previous position
        self._prev_position = self._turtle.position.copy()

        # --- Execute action ---
        move_succeeded = True
        if action == Action.FORWARD:
            old_pos = self._turtle.position.copy()
            fv = FACING_VECTORS[self._turtle.facing]
            tx = int(old_pos[0] + fv[0])
            ty = int(old_pos[1] + fv[1])
            tz = int(old_pos[2] + fv[2])
            if not self._shared_world.can_move_to(self._agent_id, tx, ty, tz):
                move_succeeded = False
            else:
                move_succeeded = self._turtle.move_forward(self._shared_world)
                if move_succeeded:
                    if not self._shared_world.move_agent(
                        self._agent_id, old_pos, self._turtle.position,
                    ):
                        self._turtle.position = old_pos.copy()
                        move_succeeded = False
        elif action == Action.UP:
            old_pos = self._turtle.position.copy()
            tx, ty, tz = int(old_pos[0]), int(old_pos[1]) + 1, int(old_pos[2])
            if not self._shared_world.can_move_to(self._agent_id, tx, ty, tz):
                move_succeeded = False
            else:
                move_succeeded = self._turtle.move_up(self._shared_world)
                if move_succeeded:
                    if not self._shared_world.move_agent(
                        self._agent_id, old_pos, self._turtle.position,
                    ):
                        self._turtle.position = old_pos.copy()
                        move_succeeded = False
        elif action == Action.DOWN:
            old_pos = self._turtle.position.copy()
            tx, ty, tz = int(old_pos[0]), int(old_pos[1]) - 1, int(old_pos[2])
            if not self._shared_world.can_move_to(self._agent_id, tx, ty, tz):
                move_succeeded = False
            else:
                move_succeeded = self._turtle.move_down(self._shared_world)
                if move_succeeded:
                    if not self._shared_world.move_agent(
                        self._agent_id, old_pos, self._turtle.position,
                    ):
                        self._turtle.position = old_pos.copy()
                        move_succeeded = False
        elif action == Action.TURN_LEFT:
            self._turtle.turn_left()
        elif action == Action.TURN_RIGHT:
            self._turtle.turn_right()
        elif action == Action.DIG:
            pre_block = self._get_dig_block(action)
            if self._turtle.dig(self._shared_world):
                block_mined = pre_block
                self._emit_block_removed(action, block_mined)
        elif action == Action.DIG_UP:
            pre_block = self._get_dig_block(action)
            if self._turtle.dig_up(self._shared_world):
                block_mined = pre_block
                self._emit_block_removed(action, block_mined)
        elif action == Action.DIG_DOWN:
            pre_block = self._get_dig_block(action)
            if self._turtle.dig_down(self._shared_world):
                block_mined = pre_block
                self._emit_block_removed(action, block_mined)

        # --- Movement failure detection ---
        if (
            action in (Action.FORWARD, Action.UP, Action.DOWN)
            and not move_succeeded
        ):
            self._emit_path_blocked(action)

        # --- Update memory and inspect ---
        px, py, pz = (
            int(self._turtle.position[0]),
            int(self._turtle.position[1]),
            int(self._turtle.position[2]),
        )
        self._memory[px, py, pz] = int(BlockType.AIR)
        self._inspected_blocks = self._inspect_three_blocks()

        # Track explored positions
        tp = (px, py, pz)
        self._explored.add(tp)

        # --- Update counters ---
        self._step_count += 1
        self._task_steps += 1
        if block_mined is not None:
            self._blocks_cleared += 1
            if int(block_mined) in _ORE_INDEX:
                ore_idx = _ORE_INDEX[int(block_mined)]
                self._steps_since_ore = 0
                self._mined_ore_counts[ore_idx] += 1
            else:
                self._steps_since_ore += 1
        else:
            self._steps_since_ore += 1

        # --- Compute reward ---
        reward = 0.0
        task_complete = False
        new_alignment = 0.0
        if self._assignment is not None:
            reward, task_complete, new_alignment = (
                compute_task_reward(
                    assignment=self._assignment,
                    agent_position=self._turtle.position,
                    prev_position=self._prev_position,
                    block_mined=block_mined,
                    blocks_cleared=self._blocks_cleared,
                    step_budget=self._assignment.step_budget,
                    task_steps=self._task_steps,
                    world_diagonal=self._world_diagonal,
                    mined_ore_counts=self._mined_ore_counts,
                    prev_alignment=self._prev_alignment,
                    **self._task_reward_config,
                )
            )
            self._prev_alignment = new_alignment

        # Termination / truncation
        terminated = False
        truncated = self._step_count >= self._max_steps

        # Send telemetry to shared world
        self._shared_world.record_telemetry(
            self._agent_id, self._step_events,
        )

        # Broadcast ore discovery
        if block_mined is not None and block_mined in _ORE_INDEX:
            ore_idx = _ORE_INDEX[block_mined]
            if self._preference[ore_idx] > 0:
                self._message_buffer.send(AgentMessage(
                    sender_id=self._agent_id,
                    message_type=MessageType.ORE_FOUND,
                    position=tp,
                    ore_type=ore_idx,
                    timestamp=self._step_count,
                ))

        obs = self._build_obs()
        info = self._build_info()
        info["block_mined"] = block_mined
        info["task_complete"] = task_complete
        info["blocks_cleared"] = self._blocks_cleared
        info["mined_ore_counts"] = self._mined_ore_counts.copy()
        info["distribution_alignment"] = new_alignment

        return obs, float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions."""
        assert self._turtle is not None
        base_mask = get_action_mask(
            self._turtle,
            self._shared_world,
            self._shared_world.shape,
        )

        # Additionally mask movement into occupied positions
        pos = self._turtle.position
        fv = FACING_VECTORS[self._turtle.facing]

        move_checks = {
            Action.FORWARD: (int(pos[0] + fv[0]), int(pos[1] + fv[1]), int(pos[2] + fv[2])),
            Action.UP: (int(pos[0]), int(pos[1]) + 1, int(pos[2])),
            Action.DOWN: (int(pos[0]), int(pos[1]) - 1, int(pos[2])),
        }

        for act, (nx, ny, nz) in move_checks.items():
            if base_mask[act] and self._shared_world.is_occupied(nx, ny, nz):
                occ = self._shared_world.get_occupant(nx, ny, nz)
                if occ != self._agent_id:
                    base_mask[act] = False

        return base_mask

    def get_astar_action(self) -> int | None:
        """Get the next A* action if in navigation mode.

        Returns None if in RL mode or no path available.
        """
        if self._assignment is None or self._turtle is None:
            return None

        # Check if we're within mining radius
        target = self._assignment.target_position
        dist = float(np.sum(np.abs(self._turtle.position - target)))
        if dist <= self._mining_radius:
            return None  # Switch to RL mode

        return self._pathfinder.get_next_action(
            self._turtle.position,
            self._turtle.facing,
            self._shared_world,
            self._agent_id,
        )

    @property
    def is_astar_mode(self) -> bool:
        """True if the agent should use A* navigation."""
        if self._assignment is None or self._turtle is None:
            return False
        target = self._assignment.target_position
        dist = float(np.sum(np.abs(self._turtle.position - target)))
        return dist > self._mining_radius

    # ------------------------------------------------------------------
    # Telemetry emission
    # ------------------------------------------------------------------

    def _emit_block_observed(
        self, x: int, y: int, z: int, block_type: int,
    ) -> None:
        """Emit a BLOCK_OBSERVED telemetry event."""
        ore_type = _ORE_INDEX.get(block_type)
        self._step_events.append(TelemetryEvent(
            event_type=TelemetryEventType.BLOCK_OBSERVED,
            agent_id=self._agent_id,
            position=(x, y, z),
            block_type=block_type,
            previous_belief=0.0,  # Tracked internally by BeliefMap
            timestamp=self._step_count,
            ore_type=ore_type,
        ))

    def _emit_block_removed(
        self, action: int, block_type: int | None,
    ) -> None:
        """Emit BLOCK_REMOVED for a mined block + dig-through BLOCK_OBSERVED."""
        dig_pos = self._get_dig_target_pos(action)
        if dig_pos is None:
            return

        ore_type = _ORE_INDEX.get(block_type) if block_type is not None else None
        self._step_events.append(TelemetryEvent(
            event_type=TelemetryEventType.BLOCK_REMOVED,
            agent_id=self._agent_id,
            position=dig_pos,
            block_type=block_type or 0,
            previous_belief=0.0,
            timestamp=self._step_count,
            ore_type=ore_type,
        ))

        # Dig-through reveal
        behind = self._dig_through_target(action, dig_pos)
        if behind is not None:
            bx, by, bz = behind
            ws = self._shared_world.shape
            if 0 <= bx < ws[0] and 0 <= by < ws[1] and 0 <= bz < ws[2]:
                block = self._shared_world.get_block(bx, by, bz)
                self._memory[bx, by, bz] = block
                self._emit_block_observed(bx, by, bz, block)

    def _emit_path_blocked(self, action: int) -> None:
        """Emit PATH_BLOCKED when a move fails unexpectedly."""
        target_pos = self._get_move_target_pos(action)
        if target_pos is None:
            return
        x, y, z = target_pos
        ws = self._shared_world.shape
        if 0 <= x < ws[0] and 0 <= y < ws[1] and 0 <= z < ws[2]:
            block = self._shared_world.get_block(x, y, z)
            self._step_events.append(TelemetryEvent(
                event_type=TelemetryEventType.PATH_BLOCKED,
                agent_id=self._agent_id,
                position=(x, y, z),
                block_type=block,
                previous_belief=0.0,
                timestamp=self._step_count,
            ))
            # Trigger A* replan
            self._pathfinder.invalidate_path()

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Build multi-agent observation dict."""
        assert self._turtle is not None

        ws = np.array(self._shared_world.shape, dtype=np.float32)
        pos = self._turtle.position.astype(np.float32)

        # --- Voxel tensor (16 channels) ---
        voxel_tensor = self._build_voxel_tensor()

        # --- Base scalars (70 dims, same layout as single-agent) ---
        norm_pos = pos / ws

        facing_oh = np.zeros(4, dtype=np.float32)
        facing_oh[self._turtle.facing] = 1.0

        fuel = np.array(
            [self._turtle.fuel / max(self._turtle.max_fuel, 1)],
            dtype=np.float32,
        )

        inv = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        for i, ore_bt in enumerate(ORE_TYPES):
            inv[i] = float(self._turtle.inventory.get(int(ore_bt), 0)) / 64.0

        world_height_norm = np.array(
            [ws[1] / MAX_WORLD_HEIGHT], dtype=np.float32,
        )

        biome_oh = np.zeros(NUM_BIOME_TYPES, dtype=np.float32)
        biome_map = self._shared_world.biome_map
        bx = min(int(pos[0]), biome_map.shape[0] - 1)
        bz = min(int(pos[2]), biome_map.shape[1] - 1)
        biome_id = int(biome_map[bx, bz])
        if 0 <= biome_id < NUM_BIOME_TYPES:
            biome_oh[biome_id] = 1.0

        # Inspection vectors (45 dims)
        front_bt, above_bt, below_bt = self._inspected_blocks
        inspect_front = self._encode_block_one_hot(front_bt)
        inspect_above = self._encode_block_one_hot(above_bt)
        inspect_below = self._encode_block_one_hot(below_bt)

        # Fog density
        mem_window = self._get_memory_window()
        fog_density = np.array(
            [np.sum(mem_window == MEMORY_UNKNOWN) / max(mem_window.size, 1)],
            dtype=np.float32,
        )

        steps_since_ore = np.array(
            [min(1.0, self._steps_since_ore / 100.0)], dtype=np.float32,
        )
        explored_frac = np.array(
            [len(self._explored) / max(ws[0] * ws[1] * ws[2], 1)],
            dtype=np.float32,
        )

        # --- Multi-agent extra scalars (10 dims) ---
        # No assignment: zeros indicate no task info
        extra = np.zeros(10, dtype=np.float32)
        if self._assignment is not None:
            target = self._assignment.target_position.astype(np.float32)
            rel_target = (target - pos) / ws
            extra[0:3] = rel_target

            dist = float(np.sum(np.abs(target - pos)))
            extra[3] = dist / max(self._world_diagonal, 1.0)

            # Task type one-hot
            tt = int(self._assignment.task_type)
            if 0 <= tt < NUM_TASK_TYPES:
                extra[4 + tt] = 1.0

            # Boundary features
            bbox = self._assignment.bounding_box
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            extra[8] = bbox.distance_to_boundary(x, y, z) / max(self._world_diagonal, 1.0)
            extra[9] = 1.0 if bbox.contains(x, y, z) else 0.0

        scalars = np.concatenate([
            norm_pos, facing_oh, fuel, inv, world_height_norm,
            biome_oh,
            inspect_front, inspect_above, inspect_below,
            fog_density, steps_since_ore, explored_frac,
            extra,
        ])

        return {
            "voxels": voxel_tensor,
            "scalars": scalars,
            "pref": self._preference.copy(),
        }

    def _build_voxel_tensor(self) -> np.ndarray:
        """Build (16, Y, X, Z) multi-channel voxel observation."""
        assert self._turtle is not None and self._memory is not None

        raw = self._get_memory_window()
        raw_yxz = raw.transpose(1, 0, 2)

        tensor = np.zeros(
            (NUM_MULTI_VOXEL_CHANNELS, FOG_WINDOW_Y, FOG_WINDOW_X, FOG_WINDOW_Z),
            dtype=np.float16,
        )

        # Channels 0-14: same as single-agent
        tensor[CH_UNKNOWN] = (raw_yxz == MEMORY_UNKNOWN).astype(np.float16)
        for i, ore_bt in enumerate(ORE_TYPES):
            tensor[1 + i] = (raw_yxz == int(ore_bt)).astype(np.float16)
        tensor[CH_SOLID] = np.isin(raw_yxz, _SOLID_ARRAY).astype(np.float16)
        tensor[CH_SOFT] = np.isin(raw_yxz, _SOFT_ARRAY).astype(np.float16)
        tensor[CH_AIR] = (raw_yxz == int(BlockType.AIR)).astype(np.float16)
        tensor[CH_BEDROCK] = (raw_yxz == int(BlockType.BEDROCK)).astype(np.float16)
        tensor[CH_EXPLORED] = self._build_explored_mask()
        # Target ore highlight
        target_mask = np.zeros_like(tensor[0])
        for i in range(NUM_ORE_TYPES):
            if self._preference[i] > 0:
                np.maximum(
                    target_mask,
                    self._preference[i] * tensor[1 + i],
                    out=target_mask,
                )
        tensor[CH_TARGET] = target_mask

        # Channel 15: Agent density
        density = self._shared_world.get_agent_density_map(
            self._turtle.position,
            radius_xz=FOG_WINDOW_RADIUS_XZ,
            y_above=FOG_WINDOW_Y_ABOVE,
            y_below=FOG_WINDOW_Y_BELOW,
            exclude_agent=self._agent_id,
        )
        # density is (X, Y, Z), need (Y, X, Z)
        tensor[CH_AGENT_DENSITY] = density.transpose(1, 0, 2).astype(np.float16)

        return tensor

    # ------------------------------------------------------------------
    # Internal helpers (from single-agent env, adapted)
    # ------------------------------------------------------------------

    def _inspect_three_blocks(self) -> tuple[int, int, int]:
        """Inspect front/above/below and update memory."""
        assert self._turtle is not None and self._memory is not None
        pos = self._turtle.position
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        ws = self._shared_world.shape

        def _inspect(x: int, y: int, z: int) -> int:
            if 0 <= x < ws[0] and 0 <= y < ws[1] and 0 <= z < ws[2]:
                block = self._shared_world.get_block(x, y, z)

                # Mismatch detection
                old_belief = self._memory[x, y, z]
                if old_belief != MEMORY_UNKNOWN and old_belief != block:
                    if old_belief == int(BlockType.AIR) and block != int(BlockType.AIR):
                        evt_type = TelemetryEventType.BLOCK_ADDED
                    else:
                        evt_type = TelemetryEventType.BLOCK_CHANGED
                    self._step_events.append(TelemetryEvent(
                        event_type=evt_type,
                        agent_id=self._agent_id,
                        position=(x, y, z),
                        block_type=block,
                        previous_belief=0.0,
                        timestamp=self._step_count,
                    ))

                self._memory[x, y, z] = block
                self._emit_block_observed(x, y, z, block)
                return block
            return int(BlockType.BEDROCK)

        fv = FACING_VECTORS[self._turtle.facing]
        front = _inspect(px + fv[0], py + fv[1], pz + fv[2])
        above = _inspect(px, py + 1, pz)
        below = _inspect(px, py - 1, pz)

        return front, above, below

    def _build_explored_mask(self) -> np.ndarray:
        """Build explored mask matching (Y, X, Z) window shape."""
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

    def _get_memory_window(self) -> np.ndarray:
        """Extract fog-of-war observation from memory."""
        assert self._turtle is not None and self._memory is not None
        pos = self._turtle.position
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        sx, sy, sz = self._memory.shape

        r = FOG_WINDOW_RADIUS_XZ
        x0, x1 = px - r, px + r + 1
        y0, y1 = py - FOG_WINDOW_Y_BELOW, py + FOG_WINDOW_Y_ABOVE + 1
        z0, z1 = pz - r, pz + r + 1

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
        """Encode a block as NUM_VOXEL_CHANNELS one-hot vector."""
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
            vec[CH_SOLID] = 1.0
        return vec

    def _get_dig_block(self, action: int) -> int | None:
        """Return block type at the dig target."""
        assert self._turtle is not None
        pos = self._turtle.position
        if action == Action.DIG:
            offset = FACING_VECTORS[self._turtle.facing]
        elif action == Action.DIG_UP:
            offset = np.array([0, 1, 0], dtype=np.int32)
        else:
            offset = np.array([0, -1, 0], dtype=np.int32)

        target = pos + offset
        ws = self._shared_world.shape
        if np.any(target < 0) or np.any(target >= np.array(ws)):
            return None
        block = self._shared_world.get_block(
            int(target[0]), int(target[1]), int(target[2]),
        )
        if block == BlockType.AIR or block == BlockType.BEDROCK:
            return None
        return block

    def _get_dig_target_pos(self, action: int) -> tuple[int, int, int] | None:
        """Return world position targeted by a dig action."""
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
        self, action: int, dig_pos: tuple[int, int, int],
    ) -> tuple[int, int, int] | None:
        """Return position behind a dug block for dig-through reveal."""
        dx, dy, dz = dig_pos
        if action == Action.DIG:
            fv = FACING_VECTORS[self._turtle.facing]
            return (dx + fv[0], dy + fv[1], dz + fv[2])
        if action == Action.DIG_UP:
            return (dx, dy + 1, dz)
        if action == Action.DIG_DOWN:
            return (dx, dy - 1, dz)
        return None

    def _get_move_target_pos(self, action: int) -> tuple[int, int, int] | None:
        """Return world position targeted by a move action."""
        assert self._turtle is not None
        pos = self._turtle.position
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        if action == Action.FORWARD:
            fv = FACING_VECTORS[self._turtle.facing]
            return (px + fv[0], py + fv[1], pz + fv[2])
        if action == Action.UP:
            return (px, py + 1, pz)
        if action == Action.DOWN:
            return (px, py - 1, pz)
        return None

    def _build_info(self) -> dict[str, Any]:
        """Build the info dict."""
        assert self._turtle is not None
        return {
            "agent_id": self._agent_id,
            "fuel": self._turtle.fuel,
            "position": self._turtle.position.copy(),
            "facing": self._turtle.facing,
            "step": self._step_count,
            "explored_count": len(self._explored),
            "is_astar_mode": self.is_astar_mode,
        }
