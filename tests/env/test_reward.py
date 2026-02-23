"""Tests for the reward system (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest
from prospect_rl.config import (
    COST_WEIGHTS,
    NUM_ORE_TYPES,
    ORE_BASE_VALUES,
    ORE_TYPES,
    REWARD_ALPHA,
    BlockType,
)
from prospect_rl.env.preference import PreferenceManager
from prospect_rl.env.reward_vector import compute_reward_vector

# ---------------------------------------------------------------------------
# Reward Vector
# ---------------------------------------------------------------------------


class _FakeTurtle:
    """Minimal turtle stub for reward tests."""
    def __init__(self, fuel: int = 100) -> None:
        self.fuel = fuel


class TestRewardVector:
    def test_ore_reward_diamond(self) -> None:
        r_ore, r_cost = compute_reward_vector(
            action=6,  # DIG
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
        )
        assert r_ore.shape == (NUM_ORE_TYPES,)
        # Diamond is at index 3 in ORE_TYPES
        diamond_idx = list(ORE_TYPES).index(BlockType.DIAMOND_ORE)
        assert r_ore[diamond_idx] == ORE_BASE_VALUES[BlockType.DIAMOND_ORE]

    def test_no_ore_all_zeros(self) -> None:
        r_ore, _ = compute_reward_vector(
            action=0,  # FORWARD
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
        )
        assert np.all(r_ore == 0)

    def test_movement_cost(self) -> None:
        _, r_cost = compute_reward_vector(
            action=0,  # FORWARD
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
        )
        np.testing.assert_almost_equal(r_cost[0], COST_WEIGHTS["movement"])
        assert r_cost[0] < 0

    def test_dig_cost(self) -> None:
        _, r_cost = compute_reward_vector(
            action=6,  # DIG
            block_mined=int(BlockType.STONE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
        )
        np.testing.assert_almost_equal(r_cost[1], COST_WEIGHTS["dig"])
        assert r_cost[1] < 0

    def test_time_penalty_always_present(self) -> None:
        _, r_cost = compute_reward_vector(
            action=4,  # TURN_LEFT
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
        )
        np.testing.assert_almost_equal(r_cost[4], COST_WEIGHTS["time_penalty"])
        assert r_cost[4] < 0

    def test_fuel_penalty_when_low(self) -> None:
        _, r_cost = compute_reward_vector(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=10),  # < 10% of 500
            max_fuel=500,
        )
        np.testing.assert_almost_equal(r_cost[2], COST_WEIGHTS["fuel_penalty"])

    def test_no_fuel_penalty_when_high(self) -> None:
        _, r_cost = compute_reward_vector(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=400),
            max_fuel=500,
        )
        assert r_cost[2] == 0.0

    def test_death_penalty_at_zero_fuel(self) -> None:
        _, r_cost = compute_reward_vector(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=0),
            max_fuel=500,
        )
        assert r_cost[3] == COST_WEIGHTS["death_penalty"]

    def test_cost_penalties_are_negative(self) -> None:
        _, r_cost = compute_reward_vector(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=0),
            max_fuel=500,
        )
        assert np.all(r_cost <= 0)


# ---------------------------------------------------------------------------
# Preference Manager
# ---------------------------------------------------------------------------

class TestPreferenceManager:
    def test_one_hot_sums_to_one(self) -> None:
        pm = PreferenceManager(seed=42)
        w = pm.sample("one_hot")
        assert w.shape == (NUM_ORE_TYPES,)
        assert abs(w.sum() - 1.0) < 1e-5
        assert np.sum(w > 0) == 1

    def test_two_mix_sums_to_one(self) -> None:
        pm = PreferenceManager(seed=42)
        w = pm.sample("two_mix")
        assert abs(w.sum() - 1.0) < 1e-5
        assert np.sum(w > 0) == 2

    def test_dirichlet_sums_to_one(self) -> None:
        pm = PreferenceManager(seed=42)
        w = pm.sample("dirichlet")
        assert abs(w.sum() - 1.0) < 1e-5
        assert np.all(w >= 0)

    def test_dirichlet_produces_varied_distributions(self) -> None:
        pm = PreferenceManager(seed=42)
        samples = [pm.sample("dirichlet") for _ in range(10)]
        # Not all the same
        assert not all(np.array_equal(samples[0], s) for s in samples[1:])

    def test_invalid_mode_raises(self) -> None:
        pm = PreferenceManager(seed=42)
        with pytest.raises(ValueError):
            pm.sample("invalid_mode")


# ---------------------------------------------------------------------------
# Scalarization
# ---------------------------------------------------------------------------

class TestScalarization:
    def test_diamond_preference_highest_for_diamond(self) -> None:
        diamond_idx = list(ORE_TYPES).index(BlockType.DIAMOND_ORE)
        w_diamond = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        w_diamond[diamond_idx] = 1.0

        w_coal = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        coal_idx = list(ORE_TYPES).index(BlockType.COAL_ORE)
        w_coal[coal_idx] = 1.0

        # Diamond mined
        r_ore = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        r_ore[diamond_idx] = ORE_BASE_VALUES[BlockType.DIAMOND_ORE]
        r_cost = np.array([-0.005, 0, 0, 0, -0.001], dtype=np.float32)

        reward_diamond_pref = PreferenceManager.scalarize(
            w_diamond, r_ore, r_cost,
        )
        reward_coal_pref = PreferenceManager.scalarize(
            w_coal, r_ore, r_cost,
        )

        assert reward_diamond_pref > reward_coal_pref

    def test_hand_calculated_scalarization(self) -> None:
        w_ore = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        w_ore[0] = 0.5  # coal
        w_ore[1] = 0.5  # iron
        r_ore = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        r_ore[0] = 1.0
        r_ore[1] = 2.0
        r_cost = np.array(
            [-0.01, -0.005, 0, 0, -0.001], dtype=np.float32,
        )

        expected = (
            REWARD_ALPHA * (0.5 * 1.0 + 0.5 * 2.0)
            + (-0.01 + -0.005 + -0.001)
        )
        actual = PreferenceManager.scalarize(w_ore, r_ore, r_cost)
        assert abs(actual - expected) < 1e-5

    def test_zero_ore_step_costs_only(self) -> None:
        w_ore = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        diamond_idx = list(ORE_TYPES).index(BlockType.DIAMOND_ORE)
        w_ore[diamond_idx] = 1.0
        r_ore = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        r_cost = np.array(
            [-0.01, 0, 0, 0, -0.001], dtype=np.float32,
        )

        reward = PreferenceManager.scalarize(w_ore, r_ore, r_cost)
        assert reward < 0  # only costs
