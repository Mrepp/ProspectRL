"""Turtle state management and action execution.

The Turtle class tracks position, facing, fuel, and inventory for the
ComputerCraft turtle agent operating within a 3D block world.
"""

from __future__ import annotations

import numpy as np
from prospect_rl.config import ACTION_FUEL_COST, Action, BlockType

# Direction vectors for each facing: 0=north(+z), 1=east(+x), 2=south(-z), 3=west(-x)
FACING_VECTORS: dict[int, np.ndarray] = {
    0: np.array([0, 0, 1]),
    1: np.array([1, 0, 0]),
    2: np.array([0, 0, -1]),
    3: np.array([-1, 0, 0]),
}


class Turtle:
    """Turtle agent with position, facing, fuel, and inventory state."""

    def __init__(
        self,
        position: np.ndarray,
        facing: int = 0,
        fuel: int = 10000,
        max_fuel: int = 10000,
    ) -> None:
        self.position = position.copy().astype(np.int32)
        self.facing = facing
        self.fuel = fuel
        self.max_fuel = max_fuel
        self.inventory: dict[int, int] = {}  # block_id -> count

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _can_move_to(self, target: np.ndarray, world: object) -> bool:
        """Check if the turtle can move to *target* position."""
        world_size = np.array(world.shape, dtype=np.int32)
        # Out of bounds
        if np.any(target < 0) or np.any(target >= world_size):
            return False
        block = world[target[0], target[1], target[2]]
        # Cannot enter bedrock
        if block == BlockType.BEDROCK:
            return False
        # Cannot walk through solid blocks (anything that isn't AIR)
        if block != BlockType.AIR:
            return False
        return True

    def _consume_fuel(self, action: int) -> bool:
        """Deduct fuel for *action*. Return False if insufficient fuel."""
        cost = ACTION_FUEL_COST.get(action, 0)
        if cost > 0 and self.fuel < cost:
            return False
        self.fuel -= cost
        return True

    def move_forward(self, world: object) -> bool:
        """Move one block in the facing direction."""
        if not self._consume_fuel(Action.FORWARD):
            return False
        target = self.position + FACING_VECTORS[self.facing]
        if not self._can_move_to(target, world):
            # Refund fuel on failure
            self.fuel += ACTION_FUEL_COST[Action.FORWARD]
            return False
        self.position = target
        return True

    def move_up(self, world: object) -> bool:
        """Move one block up (+y)."""
        if not self._consume_fuel(Action.UP):
            return False
        target = self.position + np.array([0, 1, 0], dtype=np.int32)
        if not self._can_move_to(target, world):
            self.fuel += ACTION_FUEL_COST[Action.UP]
            return False
        self.position = target
        return True

    def move_down(self, world: object) -> bool:
        """Move one block down (-y)."""
        if not self._consume_fuel(Action.DOWN):
            return False
        target = self.position + np.array([0, -1, 0], dtype=np.int32)
        if not self._can_move_to(target, world):
            self.fuel += ACTION_FUEL_COST[Action.DOWN]
            return False
        self.position = target
        return True

    # ------------------------------------------------------------------
    # Turning
    # ------------------------------------------------------------------

    def turn_left(self) -> bool:
        """Turn 90 degrees counter-clockwise. Always succeeds, no fuel cost."""
        self.facing = (self.facing - 1) % 4
        return True

    def turn_right(self) -> bool:
        """Turn 90 degrees clockwise. Always succeeds, no fuel cost."""
        self.facing = (self.facing + 1) % 4
        return True

    # ------------------------------------------------------------------
    # Digging
    # ------------------------------------------------------------------

    def _dig_at(self, target: np.ndarray, world: object) -> bool:
        """Dig block at *target*. Returns True if block was mined."""
        world_size = np.array(world.shape, dtype=np.int32)
        if np.any(target < 0) or np.any(target >= world_size):
            return False
        block = int(world[target[0], target[1], target[2]])
        # Cannot dig air or bedrock
        if block == BlockType.AIR or block == BlockType.BEDROCK:
            return False
        # Mine the block
        world[target[0], target[1], target[2]] = BlockType.AIR
        self.inventory[block] = self.inventory.get(block, 0) + 1
        return True

    def dig(self, world: object) -> bool:
        """Dig block in facing direction."""
        target = self.position + FACING_VECTORS[self.facing]
        return self._dig_at(target, world)

    def dig_up(self, world: object) -> bool:
        """Dig block above."""
        target = self.position + np.array([0, 1, 0], dtype=np.int32)
        return self._dig_at(target, world)

    def dig_down(self, world: object) -> bool:
        """Dig block below."""
        target = self.position + np.array([0, -1, 0], dtype=np.int32)
        return self._dig_at(target, world)
