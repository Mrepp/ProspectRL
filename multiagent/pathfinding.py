"""Congestion-aware A* pathfinder for multi-agent navigation.

Grid-based A* on the 3D voxel world with:
- Dig-through costs for solid blocks
- Congestion field near other agents
- 6-connected grid (no diagonals — turtles can't move diagonally)
- Action conversion for turtle movement commands
- Path caching with dynamic replanning
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

import numpy as np

from prospect_rl.config import Action, BlockType
from prospect_rl.env.turtle import FACING_VECTORS

if TYPE_CHECKING:
    from prospect_rl.multiagent.shared_world import SharedWorld


# 6-connected neighbor offsets (no diagonals)
_OFFSETS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.int32)


def _manhattan(a: np.ndarray, b: np.ndarray) -> int:
    """Manhattan distance heuristic (admissible for 6-connected grid)."""
    return int(np.sum(np.abs(a - b)))


class AStarPathfinder:
    """3D A* pathfinder with congestion-aware cost field.

    Parameters
    ----------
    dig_cost:
        Extra cost to dig through a solid block (vs 1.0 for air).
    congestion_cost:
        Extra cost per nearby agent (exponentially decayed by distance).
    congestion_radius:
        Radius in blocks for the repulsive agent field.
    max_iterations:
        Maximum A* iterations before giving up.
    """

    def __init__(
        self,
        dig_cost: float = 3.0,
        congestion_cost: float = 2.0,
        congestion_radius: int = 3,
        max_iterations: int = 5000,
    ) -> None:
        self.dig_cost = dig_cost
        self.congestion_cost = congestion_cost
        self.congestion_radius = congestion_radius
        self.max_iterations = max_iterations

        # Cached path
        self._cached_path: list[np.ndarray] | None = None
        self._cached_path_index: int = 0
        self._cached_goal: np.ndarray | None = None

        # Pending dig: when we need to dig before moving
        self._pending_dig_action: int | None = None

    def find_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        shared_world: SharedWorld,
        agent_id: int = -1,
    ) -> list[np.ndarray] | None:
        """Find a path from start to goal using A*.

        Returns a list of positions (including start), or None if no path.
        Other agents' positions are treated as impassable.
        """
        sx, sy, sz = shared_world.shape
        start_t = (int(start[0]), int(start[1]), int(start[2]))
        goal_t = (int(goal[0]), int(goal[1]), int(goal[2]))

        if start_t == goal_t:
            return [start.copy()]

        # Precompute congestion field from other agents
        agent_positions = shared_world.get_agent_positions()
        other_positions = [
            pos for aid, pos in agent_positions.items()
            if aid != agent_id
        ]

        # Priority queue: (f_cost, counter, position_tuple)
        counter = 0
        open_set: list[tuple[float, int, tuple[int, int, int]]] = []
        heapq.heappush(open_set, (0.0, counter, start_t))

        g_score: dict[tuple[int, int, int], float] = {start_t: 0.0}
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}

        start_np = np.array(start_t, dtype=np.int32)
        goal_np = np.array(goal_t, dtype=np.int32)

        iterations = 0
        while open_set and iterations < self.max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_set)

            if current == goal_t:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(np.array(node, dtype=np.int32))
                    node = came_from[node]
                path.append(np.array(start_t, dtype=np.int32))
                path.reverse()
                return path

            cx, cy, cz = current
            cur_g = g_score[current]

            for offset in _OFFSETS:
                nx = cx + offset[0]
                ny = cy + offset[1]
                nz = cz + offset[2]

                # Bounds check
                if nx < 0 or ny < 0 or nz < 0:
                    continue
                if nx >= sx or ny >= sy or nz >= sz:
                    continue

                neighbor = (nx, ny, nz)

                # Check occupancy (other agents are impassable)
                occ = shared_world.get_occupant(nx, ny, nz)
                if occ != -1 and occ != agent_id:
                    continue

                # Compute movement cost
                block = shared_world.get_block(nx, ny, nz)
                if block == BlockType.BEDROCK:
                    continue

                cost = 1.0
                if block != BlockType.AIR:
                    cost += self.dig_cost

                # Congestion cost
                neighbor_np = np.array(neighbor, dtype=np.int32)
                for other_pos in other_positions:
                    dist = float(np.sum(np.abs(neighbor_np - other_pos)))
                    if dist <= self.congestion_radius and dist > 0:
                        cost += self.congestion_cost * math.exp(
                            -dist / self.congestion_radius,
                        )

                tentative_g = cur_g + cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    h = _manhattan(neighbor_np, goal_np)
                    f = tentative_g + h
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        return None  # No path found

    def get_next_action(
        self,
        current_pos: np.ndarray,
        current_facing: int,
        shared_world: SharedWorld,
        agent_id: int = -1,
    ) -> int | None:
        """Get the next action to follow the cached path.

        Returns an ``Action`` value, or None if no path/at goal.
        Handles dig-before-move sequences automatically.
        """
        # If there's a pending dig, execute it
        if self._pending_dig_action is not None:
            action = self._pending_dig_action
            self._pending_dig_action = None
            return action

        if self._cached_path is None or self._cached_path_index >= len(self._cached_path):
            return None

        # Find current position in path
        cur = (int(current_pos[0]), int(current_pos[1]), int(current_pos[2]))
        target_idx = self._cached_path_index

        # Skip past positions we've already passed
        while target_idx < len(self._cached_path) - 1:
            wp = self._cached_path[target_idx]
            if (int(wp[0]), int(wp[1]), int(wp[2])) == cur:
                target_idx += 1
                break
            target_idx += 1

        if target_idx >= len(self._cached_path):
            return None

        self._cached_path_index = target_idx
        next_pos = self._cached_path[target_idx]
        return self._compute_action_for_step(
            current_pos, current_facing, next_pos, shared_world, agent_id,
        )

    def _compute_action_for_step(
        self,
        current_pos: np.ndarray,
        current_facing: int,
        next_pos: np.ndarray,
        shared_world: SharedWorld,
        agent_id: int,
    ) -> int:
        """Compute the action to move from current_pos toward next_pos."""
        dx = int(next_pos[0]) - int(current_pos[0])
        dy = int(next_pos[1]) - int(current_pos[1])
        dz = int(next_pos[2]) - int(current_pos[2])

        # Vertical movement
        if dy == 1:
            block_above = shared_world.get_block(
                int(current_pos[0]), int(current_pos[1]) + 1, int(current_pos[2]),
            )
            if block_above != BlockType.AIR:
                self._pending_dig_action = Action.UP
                return Action.DIG_UP
            return Action.UP

        if dy == -1:
            block_below = shared_world.get_block(
                int(current_pos[0]), int(current_pos[1]) - 1, int(current_pos[2]),
            )
            if block_below != BlockType.AIR:
                self._pending_dig_action = Action.DOWN
                return Action.DIG_DOWN
            return Action.DOWN

        # Horizontal movement: need to face the right direction first
        # Determine required facing
        required_facing = _direction_to_facing(dx, dz)
        if required_facing is None:
            # Shouldn't happen in 6-connected grid
            return Action.FORWARD

        if current_facing != required_facing:
            # Compute turn direction
            return _compute_turn(current_facing, required_facing)

        # We're facing the right way — check if we need to dig
        nx, ny, nz = int(next_pos[0]), int(next_pos[1]), int(next_pos[2])
        block_ahead = shared_world.get_block(nx, ny, nz)
        if block_ahead != BlockType.AIR:
            # Need to dig first, then move forward next step
            self._pending_dig_action = Action.FORWARD
            return Action.DIG

        return Action.FORWARD

    def set_path(self, path: list[np.ndarray], goal: np.ndarray) -> None:
        """Set a pre-computed path."""
        self._cached_path = path
        self._cached_path_index = 0
        self._cached_goal = goal.copy()
        self._pending_dig_action = None

    def invalidate_path(self) -> None:
        """Clear the cached path (e.g., on path blocked or target change)."""
        self._cached_path = None
        self._cached_path_index = 0
        self._cached_goal = None
        self._pending_dig_action = None

    def has_path(self) -> bool:
        """Check if there's a cached path."""
        return self._cached_path is not None

    def is_path_affected(self, changed_pos: tuple[int, int, int]) -> bool:
        """Check if a changed block position lies on the cached path."""
        if self._cached_path is None:
            return False
        for wp in self._cached_path[self._cached_path_index:]:
            if (int(wp[0]), int(wp[1]), int(wp[2])) == changed_pos:
                return True
        return False

    @property
    def remaining_path_length(self) -> int:
        """Number of remaining waypoints in the cached path."""
        if self._cached_path is None:
            return 0
        return max(0, len(self._cached_path) - self._cached_path_index)


def _direction_to_facing(dx: int, dz: int) -> int | None:
    """Convert a horizontal direction delta to a facing index.

    Facing: 0=+z(north), 1=+x(east), 2=-z(south), 3=-x(west)
    """
    if dz > 0:
        return 0
    if dx > 0:
        return 1
    if dz < 0:
        return 2
    if dx < 0:
        return 3
    return None


def _compute_turn(current: int, target: int) -> int:
    """Compute the optimal turn action (LEFT or RIGHT) to reach target facing."""
    diff = (target - current) % 4
    if diff == 1:
        return Action.TURN_RIGHT
    if diff == 3:
        return Action.TURN_LEFT
    # diff == 2: either direction works, pick right
    return Action.TURN_RIGHT
