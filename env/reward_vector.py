"""Per-step reward computation using potential-based harvest efficiency.

Produces separate reward components (harvest delta, adjacent penalty,
local clear bonus, operational costs) that are later scalarized by the
preference manager.

All components are scaled to roughly [-1, +1] per step for stable
VecNormalize statistics and PPO value function learning.
"""

from __future__ import annotations

import math

import numpy as np
from prospect_rl.config import (
    ORE_TYPES,
    Action,
    RewardConfig,
    Stage1RewardConfig,
)

# Pre-compute ore-type index lookup for fast reward assignment
_ORE_INDEX: dict[int, int] = {int(bt): i for i, bt in enumerate(ORE_TYPES)}

# Dig actions (6-8)
_DIG_ACTIONS = frozenset({Action.DIG, Action.DIG_UP, Action.DIG_DOWN})

_DEFAULT_CFG = RewardConfig()
_DEFAULT_S1_CFG = Stage1RewardConfig()


def _harvest_potential(
    mined: np.ndarray,
    reference_total: float,
    preference: np.ndarray,
    kappa: float,
    epsilon: float,
) -> float:
    """Compute Φ(t) = sum_i[ w_i * (1 - exp(-mined_i / (κ * ref + ε))) ].

    Parameters
    ----------
    mined:
        Shape ``(NUM_ORE_TYPES,)`` — cumulative ores mined per type.
    reference_total:
        Fixed reference ore count per type for the saturation
        denominator.  Using a fixed value (instead of per-world
        counts) eliminates episode-to-episode reward variance.
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
    denom = kappa * reference_total + epsilon
    f_values = 1.0 - np.exp(-mined / denom)
    return float(np.dot(preference, f_values))


def compute_reward_components(
    action: int,
    block_mined: int | None,
    turtle: object,
    max_fuel: int,
    preference: np.ndarray,
    reference_total: float,
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
    reference_total:
        Fixed reference ore count per type (replaces per-world counts).
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
        Potential delta + maintenance bonus for this step.
    r_adjacent:
        Adjacent ore miss penalty (negative or zero).
    r_clear:
        Local clear bonus (positive or zero).
    r_ops:
        Operational costs (fuel curve + time penalty).
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

    # --- Component 1: Harvest potential delta + maintenance bonus ---
    new_potential = _harvest_potential(
        mined_ore_counts,
        reference_total,
        preference,
        cfg.harvest_kappa,
        cfg.harvest_epsilon,
    )
    r_harvest = (
        (new_potential - prev_potential)
        + cfg.potential_maintenance_bonus * new_potential
    )

    # --- Component 2: Adjacent ore penalty (softened) ---
    raw_miss = max(0.0, adjacent_desired_weight - mined_weight)
    if raw_miss > 0.0:
        penalty_base = math.tanh(raw_miss)
        decay = 1.0 + cfg.adjacent_skip_lambda * consecutive_skip_count
        r_adjacent = -cfg.adjacent_penalty_beta * penalty_base * decay
    else:
        r_adjacent = 0.0

    # --- Update consecutive skip count (capped) ---
    if adjacent_desired_weight > 0.0 and not mined_desired:
        new_skip_count = min(
            consecutive_skip_count + 1, cfg.adjacent_skip_cap,
        )
    else:
        new_skip_count = 0

    # --- Component 3: Local clear bonus ---
    r_clear = 0.0
    if (
        prev_adjacent_weight > 0.0
        and adjacent_desired_weight_post == 0.0
    ):
        r_clear = cfg.local_clear_bonus

    # --- Component 4: Operational costs (progressive fuel curve) ---
    fuel_fraction = turtle.fuel / max(max_fuel, 1)
    if fuel_fraction < cfg.fuel_critical_threshold:
        progress = 1.0 - (fuel_fraction / cfg.fuel_critical_threshold)
        fuel_pen = cfg.fuel_critical_penalty * (progress ** 2)
    else:
        fuel_pen = 0.0
    time_pen = cfg.time_penalty
    r_ops = fuel_pen + time_pen

    return r_harvest, r_adjacent, r_clear, r_ops, new_potential, new_skip_count


def compute_stage1_reward_components(
    block_mined: int | None,
    preference: np.ndarray,
    mined_ore_counts: np.ndarray,
    cumulative_waste_count: int,
    is_new_position: bool,
    explored_count: int = 0,
    turtle_y: int = 0,
    ore_y_range: tuple[float, float] = (0.0, 39.0),
    world_height: int = 40,
    prev_nearest_target_dist: float = float("inf"),
    curr_nearest_target_dist: float = float("inf"),
    stage1_config: Stage1RewardConfig | None = None,
) -> tuple[float, float, float, float, int]:
    """Compute Stage 1 reward components.

    Stage 1 uses immediate per-ore rewards with a Y-distance
    penalty that increases the farther the turtle is from the
    target ore's spawn range.

    Returns
    -------
    r_harvest:
        Per-ore immediate reward (positive for target ore).
    r_adjacent:
        Y-distance penalty (negative when outside ore range).
    r_clear:
        Exploration bonus + approach shaping bonus.
    r_ops:
        Waste penalty (negative for non-target digs).
    new_waste_count:
        Updated cumulative waste count.
    """
    cfg = stage1_config or _DEFAULT_S1_CFG

    r_harvest = 0.0
    r_ops = cfg.time_penalty  # constant per-step cost
    new_waste_count = cumulative_waste_count

    if block_mined is not None:
        if block_mined in _ORE_INDEX:
            idx = _ORE_INDEX[block_mined]
            mined_ore_counts[idx] += 1
            if preference[idx] > 0:
                # Target ore: immediate positive reward, scaled
                # by ore-specific difficulty multiplier
                ore_mult = cfg.ore_reward_multipliers[idx]
                r_harvest = (
                    cfg.per_ore_reward
                    * float(preference[idx])
                    * ore_mult
                )
            else:
                # Non-target ore: waste with multiplier
                new_waste_count += int(
                    cfg.non_target_ore_multiplier,
                )
                ratio = min(
                    1.0, new_waste_count / cfg.waste_ramp,
                )
                r_ops += -cfg.waste_beta * ratio ** cfg.waste_alpha
        else:
            # Non-ore block (stone, dirt, etc.): waste
            new_waste_count += 1
            ratio = min(
                1.0, new_waste_count / cfg.waste_ramp,
            )
            r_ops += -cfg.waste_beta * ratio ** cfg.waste_alpha

    # Y-distance penalty: increases with distance from ore range
    y_min, y_max = ore_y_range
    if turtle_y < y_min:
        y_dist = y_min - turtle_y
    elif turtle_y > y_max:
        y_dist = turtle_y - y_max
    else:
        y_dist = 0.0
    if y_dist > 0:
        r_adjacent = (
            -cfg.y_penalty_scale * y_dist / max(world_height, 1)
        )
    else:
        # Positive bonus for being at the correct depth
        r_adjacent = cfg.y_in_range_bonus

    # Exploration bonus (progressive: strong early, decays with count)
    if is_new_position:
        halflife = max(cfg.exploration_decay_halflife, 1)
        r_clear = cfg.exploration_bonus / (
            1.0 + explored_count / halflife
        )
    else:
        r_clear = 0.0

    # Approach shaping bonus: reward for moving closer to nearest
    # visible target ore in the observation window.
    if (
        prev_nearest_target_dist < float("inf")
        or curr_nearest_target_dist < float("inf")
    ):
        # Only apply when at least one measurement is finite
        p = min(prev_nearest_target_dist, 50.0)
        c = min(curr_nearest_target_dist, 50.0)
        r_clear += cfg.approach_bonus_scale * (p - c)

    return r_harvest, r_adjacent, r_clear, r_ops, new_waste_count


def compute_stage1_terminal_bonus(
    mined_ore_counts: np.ndarray,
    preference: np.ndarray,
    total_target_ores_in_world: int,
    stage1_config: Stage1RewardConfig | None = None,
) -> float:
    """Compute end-of-episode completion bonus for Stage 1.

    Returns ``completion_scale * (target_mined / target_in_world)``.
    """
    cfg = stage1_config or _DEFAULT_S1_CFG

    if total_target_ores_in_world <= 0:
        return 0.0

    target_mined = float(
        np.dot(preference > 0, mined_ore_counts),
    )
    completion = min(
        1.0, target_mined / total_target_ores_in_world,
    )
    return cfg.completion_scale * completion
