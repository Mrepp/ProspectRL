"""Weighted task sampling and curriculum for Phase 1 training.

Generates random task assignments for single-agent task mastery
training. Supports curriculum progression from simple to complex tasks.
"""

from __future__ import annotations

import numpy as np

from prospect_rl.config import NUM_ORE_TYPES
from prospect_rl.multiagent.coordinator.assignment import (
    BoundingBox,
    TaskAssignment,
    TaskType,
)


class TaskSampler:
    """Samples random tasks for Phase 1 single-agent training.

    Parameters
    ----------
    world_size:
        (sx, sy, sz) world dimensions.
    weights:
        Sampling weights for each task type.
    seed:
        Random seed.
    """

    def __init__(
        self,
        world_size: tuple[int, int, int],
        weights: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self._world_size = world_size
        self._rng = np.random.default_rng(seed)

        default_weights = {
            "MOVE_TO": 0.35,
            "EXCAVATE": 0.40,
            "MINE_ORE": 0.20,
            "RETURN_TO": 0.05,
        }
        self._weights = weights or default_weights

        # Build probability array
        self._task_types = [
            TaskType.MOVE_TO,
            TaskType.EXCAVATE,
            TaskType.MINE_ORE,
            TaskType.RETURN_TO,
        ]
        probs = [
            self._weights.get(tt.name, 0.25)
            for tt in self._task_types
        ]
        total = sum(probs)
        self._probs = [p / total for p in probs]

        # Curriculum state
        self._difficulty = 0  # 0=easy, 1=medium, 2=hard

    def sample(
        self,
        agent_id: int = 0,
        agent_position: np.ndarray | None = None,
        preference: np.ndarray | None = None,
    ) -> TaskAssignment:
        """Sample a random task assignment."""
        sx, sy, sz = self._world_size

        # Choose task type
        task_type = self._rng.choice(self._task_types, p=self._probs)

        # Generate target position and bounding box
        if task_type == TaskType.MOVE_TO:
            return self._sample_move_to(agent_id, agent_position, preference)
        elif task_type == TaskType.EXCAVATE:
            return self._sample_excavate(agent_id, agent_position, preference)
        elif task_type == TaskType.MINE_ORE:
            return self._sample_mine_ore(agent_id, agent_position, preference)
        else:
            return self._sample_return_to(agent_id, agent_position, preference)

    def set_difficulty(self, level: int) -> None:
        """Set curriculum difficulty (0=easy, 1=medium, 2=hard)."""
        self._difficulty = max(0, min(2, level))

    def _random_position(self, margin: int = 2) -> np.ndarray:
        """Generate a random position within the world."""
        sx, sy, sz = self._world_size
        return np.array([
            int(self._rng.integers(margin, sx - margin)),
            int(self._rng.integers(margin, sy - margin)),
            int(self._rng.integers(margin, sz - margin)),
        ], dtype=np.int32)

    def _make_bbox(
        self, center: np.ndarray, half_size: int | None = None,
    ) -> BoundingBox:
        """Create a bounding box centered on a position."""
        sx, sy, sz = self._world_size

        if half_size is None:
            # Scale with difficulty
            half_size = [12, 8, 5][self._difficulty]

        x, y, z = int(center[0]), int(center[1]), int(center[2])
        return BoundingBox(
            x_min=max(0, x - half_size),
            x_max=min(sx - 1, x + half_size),
            z_min=max(0, z - half_size),
            z_max=min(sz - 1, z + half_size),
            y_min=max(0, y - half_size),
            y_max=min(sy - 1, y + half_size),
        )

    def _make_corridor_bbox(
        self, start: np.ndarray, end: np.ndarray, margin: int = 4,
    ) -> BoundingBox:
        """Bbox encompassing corridor from start to end with margin.

        Ensures agents always start inside the box for navigation tasks
        by covering the axis-aligned bounding box of both endpoints.
        """
        sx, sy, sz = self._world_size
        x0, y0, z0 = int(start[0]), int(start[1]), int(start[2])
        x1, y1, z1 = int(end[0]), int(end[1]), int(end[2])
        return BoundingBox(
            x_min=max(0, min(x0, x1) - margin),
            x_max=min(sx - 1, max(x0, x1) + margin),
            z_min=max(0, min(z0, z1) - margin),
            z_max=min(sz - 1, max(z0, z1) + margin),
            y_min=max(0, min(y0, y1) - margin),
            y_max=min(sy - 1, max(y0, y1) + margin),
        )

    def _default_preference(
        self, preference: np.ndarray | None,
    ) -> np.ndarray:
        """Return preference or sample a random one-hot."""
        if preference is not None:
            return preference.copy()
        pref = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        pref[int(self._rng.integers(0, NUM_ORE_TYPES))] = 1.0
        return pref

    def _sample_move_to(
        self, agent_id: int,
        agent_position: np.ndarray | None,
        preference: np.ndarray | None,
    ) -> TaskAssignment:
        """Sample a MOVE_TO task."""
        target = self._random_position()

        # Distance scales with difficulty
        if agent_position is not None:
            max_dist = [10, 20, 40][self._difficulty]
            direction = target.astype(np.float32) - agent_position.astype(np.float32)
            dist = float(np.linalg.norm(direction))
            if dist > max_dist and dist > 0:
                direction = direction / dist * max_dist
                target = (agent_position + direction).astype(np.int32)
                target = np.clip(target, 2, np.array(self._world_size) - 3)

        if agent_position is not None:
            bbox = self._make_corridor_bbox(agent_position, target)
        else:
            bbox = self._make_bbox(target)

        return TaskAssignment(
            agent_id=agent_id,
            task_type=TaskType.MOVE_TO,
            target_position=target,
            bounding_box=bbox,
            ore_preference=self._default_preference(preference),
        )

    def _sample_excavate(
        self, agent_id: int,
        agent_position: np.ndarray | None,
        preference: np.ndarray | None,
    ) -> TaskAssignment:
        """Sample an EXCAVATE task."""
        if agent_position is not None:
            center = agent_position.copy()
        else:
            center = self._random_position()

        bbox = self._make_bbox(center)
        budget = [20, 50, 100][self._difficulty]

        return TaskAssignment(
            agent_id=agent_id,
            task_type=TaskType.EXCAVATE,
            target_position=center,
            bounding_box=bbox,
            ore_preference=self._default_preference(preference),
            step_budget=budget,
        )

    def _dirichlet_preference(
        self, preference: np.ndarray | None,
    ) -> np.ndarray:
        """Return preference or sample a Dirichlet(0.5) vector.

        MINE_ORE tasks use continuous preferences by default so
        agents learn multi-ore distribution targeting from Phase 1.
        """
        if preference is not None:
            return preference.copy()
        raw = self._rng.dirichlet(
            np.full(NUM_ORE_TYPES, 0.5),
        ).astype(np.float32)
        return raw

    def _sample_mine_ore(
        self, agent_id: int,
        agent_position: np.ndarray | None,
        preference: np.ndarray | None,
    ) -> TaskAssignment:
        """Sample a MINE_ORE task with Dirichlet preferences."""
        if agent_position is not None:
            seed_pos = agent_position.copy()
        else:
            seed_pos = self._random_position()
        mine_half = [8, 12, 16][self._difficulty]
        bbox = self._make_bbox(seed_pos, half_size=mine_half)

        return TaskAssignment(
            agent_id=agent_id,
            task_type=TaskType.MINE_ORE,
            target_position=seed_pos,
            bounding_box=bbox,
            ore_preference=self._dirichlet_preference(preference),
            seed_position=seed_pos.copy(),
        )

    def _sample_return_to(
        self, agent_id: int,
        agent_position: np.ndarray | None,
        preference: np.ndarray | None,
    ) -> TaskAssignment:
        """Sample a RETURN_TO task."""
        sx, sy, sz = self._world_size
        # Return to world center (simulating base)
        base = np.array([sx // 2, sy // 2, sz // 2], dtype=np.int32)
        if agent_position is not None:
            bbox = self._make_corridor_bbox(agent_position, base)
        else:
            bbox = self._make_bbox(base, half_size=16)

        return TaskAssignment(
            agent_id=agent_id,
            task_type=TaskType.RETURN_TO,
            target_position=base,
            bounding_box=bbox,
            ore_preference=self._default_preference(preference),
        )
