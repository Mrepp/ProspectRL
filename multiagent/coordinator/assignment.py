"""Task assignment dataclasses for the multi-agent coordinator.

Defines task types, bounding boxes, and per-agent assignment structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class TaskType(IntEnum):
    """Types of tasks the coordinator can assign to agents."""

    MOVE_TO = 0     # navigate to position, clearing obstacles
    EXCAVATE = 1    # dig forward systematically within region
    MINE_ORE = 2    # extract ore matching preference distribution
    RETURN_TO = 3   # return to base/staging location


NUM_TASK_TYPES: int = len(TaskType)


@dataclass
class BoundingBox:
    """Spatial constraint for agent mining region.

    All tasks include a bounding box. Agent receives rewards for
    staying inside and penalties for leaving.
    """

    x_min: int
    x_max: int
    z_min: int
    z_max: int
    y_min: int = 0
    y_max: int = 255
    preference_override: np.ndarray | None = None

    def contains(self, x: int, y: int, z: int) -> bool:
        """Check if position is inside the bounding box."""
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )

    def distance_to_boundary(self, x: int, y: int, z: int) -> float:
        """Compute signed min distance to the box boundary.

        Positive inside (distance to nearest face), negative outside.
        Used as a normalized observation feature (divided by world
        diagonal).  Callers should preserve the sign — negative
        values indicate the point is outside the bounding box.
        """
        dx = min(x - self.x_min, self.x_max - x)
        dy = min(y - self.y_min, self.y_max - y)
        dz = min(z - self.z_min, self.z_max - z)
        return float(min(dx, dy, dz))

    @property
    def center(self) -> tuple[int, int, int]:
        """Return the center of the box."""
        return (
            (self.x_min + self.x_max) // 2,
            (self.y_min + self.y_max) // 2,
            (self.z_min + self.z_max) // 2,
        )

    @property
    def volume(self) -> int:
        """Return the volume of the box in blocks."""
        return (
            (self.x_max - self.x_min + 1)
            * (self.y_max - self.y_min + 1)
            * (self.z_max - self.z_min + 1)
        )


@dataclass
class TaskAssignment:
    """A task assigned to an agent by the coordinator.

    Parameters
    ----------
    agent_id:
        Which agent this assignment is for.
    task_type:
        What kind of task (MOVE_TO, EXCAVATE, MINE_ORE, RETURN_TO).
    target_position:
        Primary target position for the task.
    bounding_box:
        Spatial constraint for the task.
    ore_preference:
        8-dim preference vector for this task.
    region_index:
        Which chunk/region this assignment targets (-1 if N/A).
    step_budget:
        Maximum steps for this task (0 = unlimited).
    seed_position:
        For MINE_ORE: the position where ore was discovered.
    """

    agent_id: int
    task_type: TaskType
    target_position: np.ndarray
    bounding_box: BoundingBox
    ore_preference: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32),
    )
    region_index: int = -1
    step_budget: int = 0
    seed_position: np.ndarray | None = None

    @property
    def is_navigation_task(self) -> bool:
        """True if the agent should use A* to reach the target."""
        return self.task_type in (TaskType.MOVE_TO, TaskType.RETURN_TO)
