"""Per-step reward computation using potential-based harvest efficiency.

Produces separate reward components (harvest delta, adjacent penalty,
local clear bonus, operational costs) that are later scalarized by the
preference manager.
"""

from __future__ import annotations

import math

import numpy as np
from prospect_rl.config import (
    ORE_TYPES,
    Action,
    RewardConfig,
)

# Pre-compute ore-type index lookup for fast reward assignment
_ORE_INDEX: dict[int, int] = {int(bt): i for i, bt in enumerate(ORE_TYPES)}

# Dig actions (6-8)
_DIG_ACTIONS = frozenset({Action.DIG, Action.DIG_UP, Action.DIG_DOWN})

_DEFAULT_CFG = RewardConfig()


def _harvest_potential(
    mined: np.ndarray,
    total: np.ndarray,
    preference: np.ndarray,
    kappa: float,
    epsilon: float,
) -> float:
    """Compute Φ(t) = sum_i[ w_i * (1 - exp(-mined_i / (κ * total_i + ε))) ].

    Parameters
    ----------
    mined:
        Shape ``(NUM_ORE_TYPES,)`` — cumulative ores mined per type.
    total:
        Shape ``(NUM_ORE_TYPES,)`` — total ores per type in the world.
    preference:
        Shape ``(NUM_ORE_TYPES,)`` — preference weights (sum to 1).
    kappa:
        Saturation parameter controlling diminishing returns.
    epsilon:
        Small constant preventing division by zero.

    Returns
    -------
    Potential value in [0, 1].
    """
    denom = kappa * total + epsilon
    f_values = 1.0 - np.exp(-mined / denom)
    return float(np.dot(preference, f_values))


def compute_reward_components(
    action: int,
    block_mined: int | None,
    turtle: object,
    max_fuel: int,
    preference: np.ndarray,
    world_ore_counts: np.ndarray,
    mined_ore_counts: np.ndarray,
    prev_potential: float,
    adjacent_desired_weight: float,
    adjacent_desired_weight_post: float,
    prev_adjacent_weight: float,
    consecutive_skip_count: int,
    reward_config: RewardConfig | None = None,
) -> tuple[float, float, float, float, float, int]:
    """Compute reward components for a single step.

    Parameters
    ----------
    action:
        The action index that was executed.
    block_mined:
        The ``BlockType`` value of the mined block, or ``None``.
    turtle:
        Current ``Turtle`` instance (read ``turtle.fuel``).
    max_fuel:
        Maximum fuel capacity.
    preference:
        Shape ``(NUM_ORE_TYPES,)`` — episode preference vector.
    world_ore_counts:
        Shape ``(NUM_ORE_TYPES,)`` — total ores per type at episode start.
    mined_ore_counts:
        Shape ``(NUM_ORE_TYPES,)`` — running mined counts, **mutated in-place**.
    prev_potential:
        Φ(t-1) from the previous step.
    adjacent_desired_weight:
        Pre-action sum of preference weights for adjacent desired ores.
    adjacent_desired_weight_post:
        Post-action sum of preference weights for adjacent desired ores.
    prev_adjacent_weight:
        Adjacent desired weight from the *previous* step (for local clear).
    consecutive_skip_count:
        How many consecutive steps the turtle has been adjacent to desired
        ores without mining one.
    reward_config:
        Reward hyperparameters. Uses defaults if ``None``.

    Returns
    -------
    r_harvest:
        Potential delta for this step.
    r_adjacent:
        Adjacent ore miss penalty (negative or zero).
    r_clear:
        Local clear bonus (positive or zero).
    r_ops:
        Operational costs (fuel/death/time penalties).
    new_potential:
        Updated Φ(t) for state tracking.
    new_skip_count:
        Updated consecutive skip count.
    """
    cfg = reward_config or _DEFAULT_CFG

    # --- Update mined counts in-place ---
    mined_desired = False
    mined_weight = 0.0
    if block_mined is not None and block_mined in _ORE_INDEX:
        idx = _ORE_INDEX[block_mined]
        mined_ore_counts[idx] += 1
        if preference[idx] > 0:
            mined_desired = True
            mined_weight = float(preference[idx])

    # --- Component 1: Harvest potential delta ---
    new_potential = _harvest_potential(
        mined_ore_counts,
        world_ore_counts,
        preference,
        cfg.harvest_kappa,
        cfg.harvest_epsilon,
    )
    r_harvest = new_potential - prev_potential

    # --- Component 2: Adjacent ore penalty (softened) ---
    raw_miss = max(0.0, adjacent_desired_weight - mined_weight)
    if raw_miss > 0.0:
        penalty_base = math.tanh(raw_miss)
        decay = 1.0 + cfg.adjacent_skip_lambda * consecutive_skip_count
        r_adjacent = -cfg.adjacent_penalty_beta * penalty_base * decay
    else:
        r_adjacent = 0.0

    # --- Update consecutive skip count ---
    if adjacent_desired_weight > 0.0 and not mined_desired:
        new_skip_count = consecutive_skip_count + 1
    else:
        new_skip_count = 0

    # --- Component 3: Local clear bonus ---
    r_clear = 0.0
    if (
        prev_adjacent_weight > 0.0
        and adjacent_desired_weight_post == 0.0
    ):
        r_clear = cfg.local_clear_bonus

    # --- Component 4: Operational costs ---
    fuel_pen = (
        cfg.fuel_penalty if turtle.fuel < 0.1 * max_fuel else 0.0
    )
    death_pen = cfg.death_penalty if turtle.fuel == 0 else 0.0
    time_pen = cfg.time_penalty
    r_ops = fuel_pen + death_pen + time_pen

    return r_harvest, r_adjacent, r_clear, r_ops, new_potential, new_skip_count
