"""Tests for the potential-based harvest efficiency reward system."""

from __future__ import annotations

import numpy as np
import pytest
from prospect_rl.config import (
    NUM_ORE_TYPES,
    BlockType,
    RewardConfig,
)
from prospect_rl.env.preference import PreferenceManager
from prospect_rl.env.reward_vector import (
    _harvest_potential,
    compute_reward_components,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTurtle:
    """Minimal turtle stub for reward tests."""

    def __init__(self, fuel: int = 100) -> None:
        self.fuel = fuel


def _make_pref(index: int, weight: float = 1.0) -> np.ndarray:
    """Create a preference vector with a single nonzero entry."""
    w = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
    w[index] = weight
    return w


def _make_world_counts(**kwargs: float) -> np.ndarray:
    """Create world ore counts from keyword args like coal=100."""
    name_to_idx = {
        "coal": 0, "iron": 1, "gold": 2, "diamond": 3,
        "redstone": 4, "emerald": 5, "lapis": 6, "copper": 7,
    }
    counts = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
    for name, val in kwargs.items():
        counts[name_to_idx[name]] = val
    return counts


_CFG = RewardConfig()


# ---------------------------------------------------------------------------
# Potential-Based Harvest
# ---------------------------------------------------------------------------


class TestPotentialHarvest:
    def test_potential_increases_on_mine(self) -> None:
        pref = _make_pref(3)  # diamond
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        r_harvest, _, _, _, new_pot, _ = compute_reward_components(
            action=6,  # DIG
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r_harvest > 0.0
        assert new_pot > 0.0

    def test_potential_zero_when_no_mine(self) -> None:
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        r_harvest, _, _, _, new_pot, _ = compute_reward_components(
            action=0,  # FORWARD
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r_harvest == 0.0
        assert new_pot == 0.0

    def test_potential_diminishing_returns(self) -> None:
        """Second ore of same type gives smaller delta than first."""
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        # Mine first diamond
        r1, _, _, _, pot1, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        # Mine second diamond (mined already updated in-place)
        r2, _, _, _, pot2, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=pot1,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r2 < r1
        assert pot2 > pot1

    def test_potential_safe_when_total_zero(self) -> None:
        """Ore type with total=0 contributes 0, not NaN."""
        pref = _make_pref(3)  # diamond
        world = _make_world_counts(diamond=0)  # no diamonds in world
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        r_harvest, _, _, _, new_pot, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert not np.isnan(r_harvest)
        assert not np.isnan(new_pot)

    def test_potential_weighted_by_preference(self) -> None:
        """Higher-preference ore gives larger delta."""
        world = _make_world_counts(coal=100, diamond=100)

        # Mine coal with coal pref=0.8
        pref_coal = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        pref_coal[0] = 0.8
        pref_coal[3] = 0.2
        mined1 = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        r_coal, _, _, _, _, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.COAL_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref_coal,
            world_ore_counts=world,
            mined_ore_counts=mined1,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )

        # Mine diamond with diamond pref=0.2 (same world)
        mined2 = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        r_diamond, _, _, _, _, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref_coal,
            world_ore_counts=world,
            mined_ore_counts=mined2,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )

        assert r_coal > r_diamond

    def test_potential_bounded(self) -> None:
        """Potential is in [0, 1] since w sums to 1 and f in [0, 1]."""
        pref = _make_pref(0)
        world = _make_world_counts(coal=10)
        mined = np.array([10.0] + [0.0] * 7, dtype=np.float64)

        pot = _harvest_potential(
            mined, world, pref, _CFG.harvest_kappa, _CFG.harvest_epsilon,
        )
        assert 0.0 <= pot <= 1.0

    def test_kappa_controls_saturation(self) -> None:
        """Lower kappa means faster saturation."""
        pref = _make_pref(0)
        world = _make_world_counts(coal=100)
        mined = np.array([40.0] + [0.0] * 7, dtype=np.float64)

        pot_low_k = _harvest_potential(mined, world, pref, 0.2, 1.0)
        pot_high_k = _harvest_potential(mined, world, pref, 0.8, 1.0)
        assert pot_low_k > pot_high_k


# ---------------------------------------------------------------------------
# Adjacent Ore Penalty
# ---------------------------------------------------------------------------


class TestAdjacentPenalty:
    def test_penalty_when_adjacent_desired_ore(self) -> None:
        """Penalty fires when adjacent to desired ore, not mining."""
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_adj, _, _, _, _ = compute_reward_components(
            action=4,  # TURN_LEFT
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=1.0,  # diamond adjacent
            adjacent_desired_weight_post=1.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r_adj < 0.0

    def test_no_penalty_when_no_adjacent(self) -> None:
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_adj, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r_adj == 0.0

    def test_penalty_reduced_when_mining(self) -> None:
        """Mining a desired ore subtracts its weight from penalty."""
        pref = _make_pref(3)  # diamond w=1.0
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_adj, _, _, _, _ = compute_reward_components(
            action=6,  # DIG
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=1.0,  # 1 adjacent diamond
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        # mined_weight = 1.0, adj = 1.0 → raw_miss = 0 → no penalty
        assert r_adj == 0.0

    def test_tanh_saturates_in_dense_veins(self) -> None:
        """Penalty is bounded even with many adjacent ores."""
        pref = _make_pref(0)
        world = _make_world_counts(coal=200)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        # 5 adjacent coal ores, each w=1.0 → sum=5.0
        _, r_adj_5, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=5.0,
            adjacent_desired_weight_post=5.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        # tanh(5) ≈ 0.9999, so penalty ≈ -BETA * 1.0
        expected_max = -_CFG.adjacent_penalty_beta * 1.0
        assert r_adj_5 == pytest.approx(expected_max, abs=0.001)

    def test_skip_decay_increases_penalty(self) -> None:
        """Consecutive skips amplify penalty via lambda."""
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_skip0, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined.copy(),
            prev_potential=0.0,
            adjacent_desired_weight=1.0,
            adjacent_desired_weight_post=1.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        _, r_skip5, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined.copy(),
            prev_potential=0.0,
            adjacent_desired_weight=1.0,
            adjacent_desired_weight_post=1.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=5,
        )
        assert r_skip5 < r_skip0  # more negative

    def test_skip_count_resets_on_mine(self) -> None:
        """Skip count resets to 0 when mining a desired ore."""
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, _, _, new_skip = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=1.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=10,
        )
        assert new_skip == 0

    def test_skip_count_resets_when_no_adjacent(self) -> None:
        """Skip count resets when no adjacent desired ore."""
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, _, _, new_skip = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=10,
        )
        assert new_skip == 0

    def test_no_penalty_for_nonpreferred_ore(self) -> None:
        """Adjacent ore with w_i=0 is ignored."""
        pref = _make_pref(3)  # only diamond
        world = _make_world_counts(coal=100, diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        # adjacent_desired_weight should be 0 if only coal is adjacent
        # (coal has pref=0); env computes this, but here we pass 0
        _, r_adj, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r_adj == 0.0


# ---------------------------------------------------------------------------
# Local Clear Bonus
# ---------------------------------------------------------------------------


class TestLocalClearBonus:
    def test_bonus_when_area_cleared(self) -> None:
        """Fires when prev adjacent > 0 and post adjacent == 0."""
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=1.0,
            adjacent_desired_weight_post=0.0,  # cleared
            prev_adjacent_weight=1.0,  # prev step had ore
            consecutive_skip_count=0,
        )
        assert r_clear == _CFG.local_clear_bonus

    def test_no_bonus_when_ores_remain(self) -> None:
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=2.0,
            adjacent_desired_weight_post=1.0,  # still ores left
            prev_adjacent_weight=2.0,
            consecutive_skip_count=0,
        )
        assert r_clear == 0.0

    def test_no_bonus_when_area_was_empty(self) -> None:
        pref = _make_pref(3)
        world = _make_world_counts(diamond=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,  # was already empty
            consecutive_skip_count=0,
        )
        assert r_clear == 0.0


# ---------------------------------------------------------------------------
# Operational Costs
# ---------------------------------------------------------------------------


class TestOperationalCosts:
    def test_time_penalty_always_present(self) -> None:
        pref = _make_pref(0)
        world = _make_world_counts(coal=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, r_ops, _, _ = compute_reward_components(
            action=4,  # TURN_LEFT
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r_ops == pytest.approx(_CFG.time_penalty)

    def test_fuel_penalty_when_low(self) -> None:
        pref = _make_pref(0)
        world = _make_world_counts(coal=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, r_ops, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=10),  # < 10% of 500
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        expected = _CFG.fuel_penalty + _CFG.time_penalty
        assert r_ops == pytest.approx(expected)

    def test_death_penalty_at_zero(self) -> None:
        pref = _make_pref(0)
        world = _make_world_counts(coal=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, r_ops, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=0),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        expected = (
            _CFG.fuel_penalty + _CFG.death_penalty + _CFG.time_penalty
        )
        assert r_ops == pytest.approx(expected)

    def test_no_movement_or_dig_cost(self) -> None:
        """Movement and dig costs no longer exist."""
        pref = _make_pref(0)
        world = _make_world_counts(coal=50)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        # FORWARD action — should only have time_penalty
        _, _, _, r_ops_move, _, _ = compute_reward_components(
            action=0,  # FORWARD
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined.copy(),
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        # DIG action — should also only have time_penalty
        _, _, _, r_ops_dig, _, _ = compute_reward_components(
            action=6,  # DIG
            block_mined=int(BlockType.STONE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            world_ore_counts=world,
            mined_ore_counts=mined.copy(),
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r_ops_move == pytest.approx(_CFG.time_penalty)
        assert r_ops_dig == pytest.approx(_CFG.time_penalty)


# ---------------------------------------------------------------------------
# Scalarization
# ---------------------------------------------------------------------------


class TestScalarization:
    def test_scalarize_formula(self) -> None:
        r = PreferenceManager.scalarize(
            r_harvest=0.01,
            r_adjacent=-0.05,
            r_clear=0.2,
            r_ops=-0.001,
        )
        expected = _CFG.harvest_alpha * 0.01 + (-0.05) + 0.2 + (-0.001)
        assert r == pytest.approx(expected)

    def test_episode_bonus(self) -> None:
        bonus = PreferenceManager.compute_episode_bonus(0.7)
        expected = _CFG.episode_bonus_gamma * 0.7
        assert bonus == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Preference Manager Sampling (unchanged)
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
        assert not all(
            np.array_equal(samples[0], s) for s in samples[1:]
        )

    def test_invalid_mode_raises(self) -> None:
        pm = PreferenceManager(seed=42)
        with pytest.raises(ValueError):
            pm.sample("invalid_mode")
