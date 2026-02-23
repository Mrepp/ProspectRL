"""Per-step vectorized reward computation.

Produces separate ore-reward and cost-penalty vectors that are later
scalarized by the preference manager.
"""

from __future__ import annotations

import numpy as np
from prospect_rl.config import (
    COST_WEIGHTS,
    NUM_ORE_TYPES,
    ORE_BASE_VALUES,
    ORE_TYPES,
    Action,
)

# Pre-compute ore-type index lookup for fast reward assignment
_ORE_INDEX: dict[int, int] = {int(bt): i for i, bt in enumerate(ORE_TYPES)}

# Movement actions (0-3), dig actions (6-8)
_MOVE_ACTIONS = frozenset({Action.FORWARD, Action.BACK, Action.UP, Action.DOWN})
_DIG_ACTIONS = frozenset({Action.DIG, Action.DIG_UP, Action.DIG_DOWN})


def compute_reward_vector(
    action: int,
    block_mined: int | None,
    turtle: object,
    max_fuel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the decomposed reward for a single step.

    Parameters
    ----------
    action:
        The action index that was executed.
    block_mined:
        The ``BlockType`` value of the mined block, or ``None`` if no block
        was mined this step.
    turtle:
        Current ``Turtle`` instance (read ``turtle.fuel``).
    max_fuel:
        Maximum fuel capacity (used for low-fuel threshold).

    Returns
    -------
    r_ore:
        Shape ``(NUM_ORE_TYPES,)`` — reward contribution per ore type.
    r_cost:
        Shape ``(5,)`` — ``[movement, dig, fuel_penalty, death_penalty, time_penalty]``.
    """
    # --- Ore reward ---
    r_ore = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
    if block_mined is not None and block_mined in _ORE_INDEX:
        idx = _ORE_INDEX[block_mined]
        r_ore[idx] = ORE_BASE_VALUES[block_mined]

    # --- Cost vector ---
    movement_cost = COST_WEIGHTS["movement"] if action in _MOVE_ACTIONS else 0.0
    dig_cost = COST_WEIGHTS["dig"] if action in _DIG_ACTIONS else 0.0
    fuel_penalty = (
        COST_WEIGHTS["fuel_penalty"]
        if turtle.fuel < 0.1 * max_fuel
        else 0.0
    )
    death_penalty = (
        COST_WEIGHTS["death_penalty"] if turtle.fuel == 0 else 0.0
    )
    time_penalty = COST_WEIGHTS["time_penalty"]

    r_cost = np.array(
        [movement_cost, dig_cost, fuel_penalty, death_penalty, time_penalty],
        dtype=np.float32,
    )

    return r_ore, r_cost
