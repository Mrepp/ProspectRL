"""Action masking for valid turtle actions.

Returns a boolean mask indicating which of the 9 discrete actions are
currently legal given the turtle state and surrounding world blocks.
"""

from __future__ import annotations

import numpy as np
from prospect_rl.config import ACTION_FUEL_COST, Action, BlockType
from prospect_rl.env.turtle import FACING_VECTORS


def _target_block(
    position: np.ndarray,
    offset: np.ndarray,
    world: object,
) -> int | None:
    """Return the block type at position+offset, or None if out of bounds."""
    target = position + offset
    world_size = np.array(world.shape, dtype=np.int32)
    if np.any(target < 0) or np.any(target >= world_size):
        return None
    return int(world[target[0], target[1], target[2]])


def get_action_mask(
    turtle: object,
    world: object,
    world_size: tuple[int, int, int],
) -> np.ndarray:
    """Return a bool array of shape ``(9,)`` where True means the action is valid."""
    mask = np.ones(9, dtype=bool)
    pos = turtle.position
    facing_vec = FACING_VECTORS[turtle.facing]

    up = np.array([0, 1, 0], dtype=np.int32)
    down = np.array([0, -1, 0], dtype=np.int32)

    # --- Movement actions (0-3) ---
    move_offsets = {
        Action.FORWARD: facing_vec,
        Action.BACK: -facing_vec,
        Action.UP: up,
        Action.DOWN: down,
    }

    for action, offset in move_offsets.items():
        cost = ACTION_FUEL_COST[action]
        if cost > 0 and turtle.fuel < cost:
            mask[action] = False
            continue
        block = _target_block(pos, offset, world)
        if block is None:
            # Out of bounds
            mask[action] = False
        elif block == BlockType.BEDROCK:
            mask[action] = False
        elif block != BlockType.AIR:
            # Solid block — must dig first
            mask[action] = False

    # --- Turns (4-5) are always valid ---
    # mask[Action.TURN_LEFT] = True  (already True)
    # mask[Action.TURN_RIGHT] = True

    # --- Dig actions (6-8) ---
    dig_offsets = {
        Action.DIG: facing_vec,
        Action.DIG_UP: up,
        Action.DIG_DOWN: down,
    }

    for action, offset in dig_offsets.items():
        block = _target_block(pos, offset, world)
        if block is None:
            mask[action] = False
        elif block == BlockType.AIR or block == BlockType.BEDROCK:
            mask[action] = False

    return mask
