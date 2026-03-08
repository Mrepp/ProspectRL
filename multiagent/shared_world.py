"""Shared world with occupancy grid for multi-agent coordination.

Wraps the existing ``World`` class and adds:
- Agent occupancy tracking (O(1) collision checks)
- Telemetry buffering for coordinator consumption
- Multi-agent spawn with minimum distance constraints
- Agent density channel for observation
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from prospect_rl.config import BlockType, ORE_TYPES

if TYPE_CHECKING:
    from prospect_rl.env.turtle import Turtle
    from prospect_rl.multiagent.telemetry import TelemetryEvent


class SharedWorld:
    """World wrapper with occupancy grid and telemetry buffer.

    Parameters
    ----------
    world:
        The underlying ``World`` instance (procedural or real-chunk).
    max_agents:
        Maximum number of agents supported.
    """

    def __init__(self, world: object, max_agents: int = 100) -> None:
        self._world = world
        self._max_agents = max_agents

        sx, sy, sz = world.shape
        self._size: tuple[int, int, int] = (sx, sy, sz)

        # Occupancy grid: -1 = empty, otherwise agent_id
        self._occupancy = np.full((sx, sy, sz), -1, dtype=np.int16)

        # Registered agents: agent_id -> Turtle
        self._agents: dict[int, Turtle] = {}

        # Telemetry buffer: agent_id -> list of events since last flush
        self._telemetry_buffer: dict[int, list[TelemetryEvent]] = defaultdict(list)

        # Step counter for telemetry timestamps
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Properties (delegate to underlying world)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._size

    @property
    def biome_map(self) -> np.ndarray:
        return self._world.biome_map

    @property
    def world(self) -> object:
        """Access the underlying World object."""
        return self._world

    @property
    def global_step(self) -> int:
        return self._global_step

    def increment_step(self) -> None:
        self._global_step += 1

    # ------------------------------------------------------------------
    # Block access (delegate to underlying world)
    # ------------------------------------------------------------------

    def get_block(self, x: int, y: int, z: int) -> int:
        """Return the block type at (x, y, z)."""
        return int(self._world[x, y, z])

    def set_block(self, x: int, y: int, z: int, block_id: int) -> None:
        """Set the block at (x, y, z)."""
        self._world[x, y, z] = np.int8(block_id)

    def count_blocks(self, block_ids: list[int]) -> int:
        """Count total blocks matching any of the given IDs."""
        return self._world.count_blocks(block_ids)

    def __getitem__(self, key):
        return self._world[key]

    def __setitem__(self, key, value):
        self._world[key] = value

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: int, turtle: Turtle) -> bool:
        """Register an agent and mark its position in the occupancy grid.

        Returns False if position is already occupied.
        """
        if agent_id in self._agents:
            return False
        if len(self._agents) >= self._max_agents:
            return False

        pos = turtle.position
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        if self._occupancy[x, y, z] != -1:
            return False

        self._agents[agent_id] = turtle
        self._occupancy[x, y, z] = agent_id
        return True

    def deregister_agent(self, agent_id: int) -> None:
        """Remove an agent from the world."""
        if agent_id not in self._agents:
            return
        turtle = self._agents[agent_id]
        pos = turtle.position
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        if self._occupancy[x, y, z] == agent_id:
            self._occupancy[x, y, z] = -1
        del self._agents[agent_id]
        self._telemetry_buffer.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Movement with occupancy
    # ------------------------------------------------------------------

    def can_move_to(self, agent_id: int, x: int, y: int, z: int) -> bool:
        """Check if an agent can move to (x, y, z).

        Position must be AIR and not occupied by another agent.
        """
        sx, sy, sz = self._size
        if x < 0 or y < 0 or z < 0:
            return False
        if x >= sx or y >= sy or z >= sz:
            return False
        block = int(self._world[x, y, z])
        if block != BlockType.AIR:
            return False
        occ = self._occupancy[x, y, z]
        return occ == -1 or occ == agent_id

    def move_agent(
        self, agent_id: int, old_pos: np.ndarray, new_pos: np.ndarray,
    ) -> bool:
        """Atomically update occupancy for an agent move.

        Returns False if the new position is not available.
        """
        nx, ny, nz = int(new_pos[0]), int(new_pos[1]), int(new_pos[2])
        if not self.can_move_to(agent_id, nx, ny, nz):
            return False

        ox, oy, oz = int(old_pos[0]), int(old_pos[1]), int(old_pos[2])
        if self._occupancy[ox, oy, oz] == agent_id:
            self._occupancy[ox, oy, oz] = -1
        self._occupancy[nx, ny, nz] = agent_id
        return True

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """Check if a position is occupied by any agent."""
        sx, sy, sz = self._size
        if x < 0 or y < 0 or z < 0:
            return False
        if x >= sx or y >= sy or z >= sz:
            return False
        return self._occupancy[x, y, z] != -1

    def get_occupant(self, x: int, y: int, z: int) -> int:
        """Return the agent_id at (x, y, z), or -1 if empty."""
        sx, sy, sz = self._size
        if x < 0 or y < 0 or z < 0:
            return -1
        if x >= sx or y >= sy or z >= sz:
            return -1
        return int(self._occupancy[x, y, z])

    # ------------------------------------------------------------------
    # Agent queries
    # ------------------------------------------------------------------

    def get_agent_positions(self) -> dict[int, np.ndarray]:
        """Return current positions of all registered agents."""
        return {
            aid: turtle.position.copy()
            for aid, turtle in self._agents.items()
        }

    def get_num_agents(self) -> int:
        return len(self._agents)

    def get_agent(self, agent_id: int) -> Turtle | None:
        return self._agents.get(agent_id)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def record_telemetry(
        self, agent_id: int, events: list[TelemetryEvent],
    ) -> None:
        """Buffer telemetry events from an agent."""
        self._telemetry_buffer[agent_id].extend(events)

    def flush_telemetry(self) -> dict[int, list[TelemetryEvent]]:
        """Return and clear all buffered telemetry events.

        Returns a dict mapping agent_id to list of events.
        """
        result = {aid: list(events) for aid, events in self._telemetry_buffer.items()}
        self._telemetry_buffer = defaultdict(list)
        return result

    def get_pending_telemetry(self) -> list[TelemetryEvent]:
        """Return all pending events as a flat list (does not clear)."""
        events = []
        for evts in self._telemetry_buffer.values():
            events.extend(evts)
        return events

    # ------------------------------------------------------------------
    # Agent spawning
    # ------------------------------------------------------------------

    def spawn_positions(
        self,
        n: int,
        min_distance: int = 5,
        rng: np.random.Generator | None = None,
    ) -> list[np.ndarray]:
        """Generate spawn positions with minimum Manhattan distance.

        Prefers existing AIR positions.  Falls back to clearing non-ore
        solid blocks.  Never destroys ore blocks.
        """
        if rng is None:
            rng = np.random.default_rng()

        ore_set = {int(bt) for bt in ORE_TYPES}
        sx, sy, sz = self._size
        positions: list[np.ndarray] = []
        max_attempts = n * 50

        for _ in range(max_attempts):
            if len(positions) >= n:
                break
            x = int(rng.integers(1, sx - 1))
            y = int(rng.integers(1, sy - 1))
            z = int(rng.integers(1, sz - 1))

            block = int(self._world[x, y, z])
            if block == BlockType.BEDROCK:
                continue
            # Never destroy ore blocks for spawn positions
            if block in ore_set:
                continue

            # Check minimum distance to existing spawns
            pos = np.array([x, y, z], dtype=np.int32)
            too_close = False
            for existing in positions:
                dist = int(np.sum(np.abs(pos - existing)))
                if dist < min_distance:
                    too_close = True
                    break
            if too_close:
                continue

            # Clear non-ore block for agent spawn
            if block != BlockType.AIR:
                self._world[x, y, z] = BlockType.AIR
            positions.append(pos)

        return positions

    # ------------------------------------------------------------------
    # Agent density channel (for multi-agent observation)
    # ------------------------------------------------------------------

    def get_agent_density_map(
        self,
        center: np.ndarray,
        radius_xz: int = 3,
        y_above: int = 3,
        y_below: int = 7,
        exclude_agent: int = -1,
    ) -> np.ndarray:
        """Extract a local agent density map as a float array.

        Returns shape ``(window_x, window_y, window_z)`` with 1.0
        where another agent is present, 0.0 otherwise.
        """
        px, py, pz = int(center[0]), int(center[1]), int(center[2])
        sx, sy, sz = self._size
        wx = 2 * radius_xz + 1
        wy = y_above + 1 + y_below
        wz = 2 * radius_xz + 1

        density = np.zeros((wx, wy, wz), dtype=np.float32)

        x0, x1 = px - radius_xz, px + radius_xz + 1
        y0, y1 = py - y_below, py + y_above + 1
        z0, z1 = pz - radius_xz, pz + radius_xz + 1

        # Clamp to world bounds
        src_x0, src_x1 = max(0, x0), min(sx, x1)
        src_y0, src_y1 = max(0, y0), min(sy, y1)
        src_z0, src_z1 = max(0, z0), min(sz, z1)

        if src_x0 >= src_x1 or src_y0 >= src_y1 or src_z0 >= src_z1:
            return density

        occ_slice = self._occupancy[
            src_x0:src_x1, src_y0:src_y1, src_z0:src_z1,
        ]

        # Offset into density array
        dx0, dy0, dz0 = src_x0 - x0, src_y0 - y0, src_z0 - z0
        dx1, dy1, dz1 = dx0 + (src_x1 - src_x0), dy0 + (src_y1 - src_y0), dz0 + (src_z1 - src_z0)

        occupied = (occ_slice != -1)
        if exclude_agent >= 0:
            occupied = occupied & (occ_slice != exclude_agent)
        density[dx0:dx1, dy0:dy1, dz0:dz1] = occupied.astype(np.float32)

        return density
