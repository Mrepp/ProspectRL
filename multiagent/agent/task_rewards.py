"""Task-specific reward computation for multi-agent mining.

All tasks share a common reward structure with task-specific
completion conditions and boundary shaping.

MINE_ORE uses distribution-aware rewards: per-block weighted by
preference * rarity, plus a potential-based alignment bonus that
nudges the agent toward its target ore distribution.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from prospect_rl.config import NUM_ORE_TYPES, ORE_INDEX as _ORE_INDEX, ORE_TYPES
from prospect_rl.multiagent.coordinator.assignment import (
    BoundingBox,
    TaskAssignment,
    TaskType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SpawnRateNormalizer — derives rarity multipliers from worldgen data
# ---------------------------------------------------------------------------

_NORMALIZER: "SpawnRateNormalizer | None" = None


class SpawnRateNormalizer:
    """Derives per-ore rarity multipliers from vanilla MC worldgen data.

    Rarity multipliers make 1 diamond "count" as much as N coal in the
    distribution alignment computation.  Derived from expected blocks
    per chunk across all placed features.

    Parameters
    ----------
    rarity_mult_cap:
        Maximum rarity multiplier (default 5.0).
    """

    def __init__(self, rarity_mult_cap: float = 5.0) -> None:
        self._rarity_mult = np.ones(NUM_ORE_TYPES, dtype=np.float32)
        self._rarity_mult_cap = rarity_mult_cap
        self._expected_per_chunk = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        self._initialized = False

    @property
    def rarity_mult(self) -> np.ndarray:
        """Per-ore rarity multipliers, shape (8,)."""
        if not self._initialized:
            self._compute()
        return self._rarity_mult

    @property
    def expected_per_chunk(self) -> np.ndarray:
        """Expected ore blocks per chunk per ore type, shape (8,)."""
        if not self._initialized:
            self._compute()
        return self._expected_per_chunk

    def _compute(self) -> None:
        """Compute rarity multipliers from worldgen parser data."""
        self._initialized = True
        try:
            from prospect_rl.env.worldgen_parser import (
                TRACKED_ORE_BLOCKS,
                load_worldgen,
            )
            from prospect_rl.multiagent.geological_prior import _expected_blocks
        except Exception:
            logger.warning("SpawnRateNormalizer: worldgen data unavailable, using uniform rarity")
            return

        try:
            wg = load_worldgen()
        except Exception:
            logger.warning("SpawnRateNormalizer: failed to load worldgen JSON, using fallback")
            self._compute_fallback()
            return

        # Map canonical ore names to our 0-7 indices
        canonical_to_idx: dict[str, int] = {}
        for i, bt in enumerate(ORE_TYPES):
            name = bt.name.lower().replace("_ore", "")
            canonical_to_idx[name] = i

        # Sum expected_attempts * expected_blocks(size) per ore type
        # across all placed features (biome-averaged)
        expected = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        for pf in wg.placed_features.values():
            if pf.configured is None:
                continue
            ore_block = pf.configured.ore_block
            if ore_block not in TRACKED_ORE_BLOCKS:
                continue
            canonical = TRACKED_ORE_BLOCKS[ore_block]
            if canonical not in canonical_to_idx:
                continue
            idx = canonical_to_idx[canonical]
            blocks = _expected_blocks(pf.configured.size)
            attempts = pf.attempt_model.expected_attempts
            expected[idx] += attempts * blocks

        self._expected_per_chunk = expected.copy()

        # Compute rarity multipliers: median / expected[i]
        nonzero = expected[expected > 0]
        if len(nonzero) == 0:
            return

        median_expected = float(np.median(nonzero))
        for i in range(NUM_ORE_TYPES):
            if expected[i] > 0:
                raw = median_expected / expected[i]
                self._rarity_mult[i] = min(raw, self._rarity_mult_cap)
            else:
                self._rarity_mult[i] = self._rarity_mult_cap

    def _compute_fallback(self) -> None:
        """Fallback: use typical_vein_size from OreTypeConfig."""
        try:
            from prospect_rl.config import ORE_TYPE_CONFIGS
            from prospect_rl.multiagent.geological_prior import _expected_blocks
        except Exception:
            return

        expected = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        for i, cfg in enumerate(ORE_TYPE_CONFIGS):
            expected[i] = _expected_blocks(cfg.typical_vein_size)

        self._expected_per_chunk = expected.copy()
        nonzero = expected[expected > 0]
        if len(nonzero) == 0:
            return

        median_expected = float(np.median(nonzero))
        for i in range(NUM_ORE_TYPES):
            if expected[i] > 0:
                raw = median_expected / expected[i]
                self._rarity_mult[i] = min(raw, self._rarity_mult_cap)
            else:
                self._rarity_mult[i] = self._rarity_mult_cap


def _get_normalizer() -> SpawnRateNormalizer:
    """Return the module-level SpawnRateNormalizer singleton."""
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = SpawnRateNormalizer()
    return _NORMALIZER


def _set_normalizer(normalizer: SpawnRateNormalizer) -> None:
    """Override the module-level normalizer (for testing)."""
    global _NORMALIZER
    _NORMALIZER = normalizer


# ---------------------------------------------------------------------------
# Distribution alignment computation
# ---------------------------------------------------------------------------


def _compute_alignment(
    mined_ore_counts: np.ndarray,
    preference: np.ndarray,
    rarity_mult: np.ndarray,
) -> float:
    """Compute cosine similarity between rarity-weighted mining and preference.

    Returns 0.0 if fewer than 1 ore block has been mined.
    """
    total_mined = float(np.sum(mined_ore_counts))
    if total_mined < 1.0:
        return 0.0

    # Rarity-weighted mined distribution
    weighted = mined_ore_counts.astype(np.float64) * rarity_mult
    weighted_sum = float(np.sum(weighted))
    if weighted_sum <= 0:
        return 0.0
    actual = weighted / weighted_sum

    # Normalize preference
    pref_sum = float(np.sum(preference))
    if pref_sum <= 0:
        return 0.0
    target = preference.astype(np.float64) / pref_sum

    # Cosine similarity
    dot = float(np.dot(actual, target))
    norm_a = float(np.linalg.norm(actual))
    norm_t = float(np.linalg.norm(target))
    if norm_a <= 0 or norm_t <= 0:
        return 0.0
    return dot / (norm_a * norm_t)


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------


def compute_task_reward(
    assignment: TaskAssignment,
    agent_position: np.ndarray,
    prev_position: np.ndarray,
    block_mined: int | None,
    blocks_cleared: int,
    step_budget: int,
    task_steps: int,
    world_diagonal: float,
    # Reward scales (from MultiAgentConfig)
    completion_reward: float = 1.0,
    progress_reward: float = 0.1,
    block_cleared_reward: float = 0.05,
    regress_penalty: float = -0.02,
    idle_penalty: float = -0.01,
    box_stay_bonus: float = 0.05,
    box_leave_penalty: float = -0.2,
    # Distribution-aware MINE_ORE params
    mined_ore_counts: np.ndarray | None = None,
    prev_alignment: float = 0.0,
    alignment_bonus: float = 0.05,
    rarity_mult_cap: float = 5.0,
    alignment_min_ores: int = 3,
) -> tuple[float, bool, float]:
    """Compute reward for the current step given a task assignment.

    Returns
    -------
    reward:
        Scalar reward for this step.
    task_complete:
        True if the task completion condition is met.
    new_alignment:
        Updated distribution alignment (only meaningful for MINE_ORE).
    """
    reward = 0.0
    task_complete = False
    new_alignment = 0.0
    bbox = assignment.bounding_box
    pos = agent_position
    x, y, z = int(pos[0]), int(pos[1]), int(pos[2])

    # --- Boundary shaping ---
    if bbox.contains(x, y, z):
        if block_mined is not None:
            reward += box_stay_bonus
    else:
        reward += box_leave_penalty

    # --- Task-specific rewards ---
    if assignment.task_type == TaskType.MOVE_TO:
        reward_delta, task_complete = _reward_move_to(
            assignment, agent_position, prev_position,
            world_diagonal, progress_reward, regress_penalty,
            completion_reward,
        )
        reward += reward_delta

    elif assignment.task_type == TaskType.EXCAVATE:
        reward_delta, task_complete = _reward_excavate(
            assignment, block_mined, blocks_cleared,
            step_budget, task_steps, bbox,
            block_cleared_reward, completion_reward, idle_penalty,
        )
        reward += reward_delta

    elif assignment.task_type == TaskType.MINE_ORE:
        reward_delta, task_complete, new_alignment = _reward_mine_ore(
            assignment, block_mined, agent_position,
            progress_reward, idle_penalty,
            mined_ore_counts=mined_ore_counts,
            prev_alignment=prev_alignment,
            alignment_bonus=alignment_bonus,
            alignment_min_ores=alignment_min_ores,
        )
        reward += reward_delta

    elif assignment.task_type == TaskType.RETURN_TO:
        reward_delta, task_complete = _reward_return_to(
            assignment, agent_position, prev_position,
            world_diagonal, progress_reward, regress_penalty,
            completion_reward,
        )
        reward += reward_delta

    return reward, task_complete, new_alignment


def _reward_move_to(
    assignment: TaskAssignment,
    agent_position: np.ndarray,
    prev_position: np.ndarray,
    world_diagonal: float,  # unused after flat-reward change
    progress_reward: float,
    regress_penalty: float,
    completion_reward: float,
) -> tuple[float, bool]:
    """MOVE_TO: Navigate to target position."""
    target = assignment.target_position
    curr_dist = float(np.sum(np.abs(agent_position - target)))
    prev_dist = float(np.sum(np.abs(prev_position - target)))

    reward = 0.0

    # Progress: flat per-block reward for closing Manhattan distance
    delta = prev_dist - curr_dist  # +1 per block closer, -1 per block farther
    if delta > 0:
        reward += progress_reward  # 0.1 per block of progress
    elif delta < 0:
        reward += regress_penalty

    # Completion: within 1 block of target
    complete = curr_dist <= 1.0
    if complete:
        reward += completion_reward

    return reward, complete


def _reward_excavate(
    assignment: TaskAssignment,
    block_mined: int | None,
    blocks_cleared: int,
    step_budget: int,
    task_steps: int,
    bbox: BoundingBox,
    block_cleared_reward: float,
    completion_reward: float,
    idle_penalty: float,
) -> tuple[float, bool]:
    """EXCAVATE: Dig forward systematically within region."""
    step_budget = max(step_budget, 1)
    reward = 0.0

    if block_mined is not None:
        reward += block_cleared_reward
        # Extra reward for mining target ore
        if block_mined in _ORE_INDEX:
            ore_idx = _ORE_INDEX[block_mined]
            if assignment.ore_preference[ore_idx] > 0:
                reward += block_cleared_reward * 3.0
    else:
        reward += idle_penalty

    # Completion: blocks cleared >= budget or budget elapsed
    complete = False
    if step_budget > 0:
        if blocks_cleared >= step_budget:
            complete = True
            reward += completion_reward
        elif task_steps >= step_budget * 2:
            # Timed out
            complete = True

    return reward, complete


def _reward_mine_ore(
    assignment: TaskAssignment,
    block_mined: int | None,
    agent_position: np.ndarray,
    progress_reward: float,
    idle_penalty: float,
    *,
    mined_ore_counts: np.ndarray | None = None,
    prev_alignment: float = 0.0,
    alignment_bonus: float = 0.05,
    alignment_min_ores: int = 3,
) -> tuple[float, bool, float]:
    """MINE_ORE: Extract ore matching preference distribution.

    Per-block reward weighted by preference * rarity, plus a
    potential-based distribution alignment bonus.

    Returns (reward, task_complete, new_alignment).
    """
    normalizer = _get_normalizer()
    rarity_mult = normalizer.rarity_mult
    preference = assignment.ore_preference
    reward = 0.0

    if block_mined is not None:
        if block_mined in _ORE_INDEX:
            ore_idx = _ORE_INDEX[block_mined]
            if preference[ore_idx] > 0:
                # Target ore: weighted by preference * rarity
                weight = min(preference[ore_idx] * rarity_mult[ore_idx], 3.0)
                reward += progress_reward * 2.0 * weight
            else:
                # Non-target ore: tolerant — still positive
                reward += progress_reward * 0.3
        else:
            # Non-ore block: mild penalty
            reward += idle_penalty * 0.5
    else:
        reward += idle_penalty

    # --- Distribution alignment bonus (potential-based) ---
    new_alignment = prev_alignment
    if mined_ore_counts is not None:
        total_mined = int(np.sum(mined_ore_counts))
        if total_mined >= alignment_min_ores:
            new_alignment = _compute_alignment(
                mined_ore_counts, preference, rarity_mult,
            )
            alignment_delta = new_alignment - prev_alignment
            reward += alignment_bonus * alignment_delta

    # Completion is determined externally — coordinator decides
    return reward, False, new_alignment


def _reward_return_to(
    assignment: TaskAssignment,
    agent_position: np.ndarray,
    prev_position: np.ndarray,
    world_diagonal: float,
    progress_reward: float,
    regress_penalty: float,
    completion_reward: float,
) -> tuple[float, bool]:
    """RETURN_TO: Return to base/staging location."""
    # Same as MOVE_TO
    return _reward_move_to(
        assignment, agent_position, prev_position,
        world_diagonal, progress_reward, regress_penalty,
        completion_reward,
    )
