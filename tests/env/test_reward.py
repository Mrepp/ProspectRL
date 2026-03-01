"""Tests for the potential-based harvest efficiency reward system."""

from __future__ import annotations

import numpy as np
import pytest
from prospect_rl.config import (
    NUM_ORE_TYPES,
    BlockType,
    RewardConfig,
    Stage1RewardConfig,
)
from prospect_rl.env.preference import PreferenceManager
from prospect_rl.env.reward_vector import (
    _harvest_potential,
    compute_reward_components,
    compute_stage1_reward_components,
    compute_stage1_terminal_bonus,
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


_CFG = RewardConfig()
_REF_TOTAL = _CFG.harvest_reference_total


# ---------------------------------------------------------------------------
# Potential-Based Harvest
# ---------------------------------------------------------------------------


class TestPotentialHarvest:
    def test_potential_increases_on_mine(self) -> None:
        pref = _make_pref(3)  # diamond
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        r_harvest, _, _, _, new_pot, _ = compute_reward_components(
            action=6,  # DIG
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        r_harvest, _, _, _, new_pot, _ = compute_reward_components(
            action=0,  # FORWARD
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        # Mine first diamond
        r1, _, _, _, pot1, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined,
            prev_potential=pot1,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r2 < r1
        assert pot2 > pot1

    def test_potential_safe_when_reference_small(self) -> None:
        """Small reference total does not produce NaN."""
        pref = _make_pref(3)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        r_harvest, _, _, _, new_pot, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=0.0,
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
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined1,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )

        # Mine diamond with diamond pref=0.2 (same pref vector)
        mined2 = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        r_diamond, _, _, _, _, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref_coal,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined2,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )

        assert r_coal > r_diamond

    def test_potential_bounded(self) -> None:
        """Potential is in [0, 1] since w sums to 1 and f in [0,1]."""
        pref = _make_pref(0)
        mined = np.array(
            [10.0] + [0.0] * 7, dtype=np.float64,
        )

        pot = _harvest_potential(
            mined, _REF_TOTAL, pref,
            _CFG.harvest_kappa, _CFG.harvest_epsilon,
        )
        assert 0.0 <= pot <= 1.0

    def test_kappa_controls_saturation(self) -> None:
        """Lower kappa means faster saturation."""
        pref = _make_pref(0)
        mined = np.array(
            [40.0] + [0.0] * 7, dtype=np.float64,
        )

        pot_low_k = _harvest_potential(
            mined, _REF_TOTAL, pref, 0.2, 1.0,
        )
        pot_high_k = _harvest_potential(
            mined, _REF_TOTAL, pref, 0.8, 1.0,
        )
        assert pot_low_k > pot_high_k

    def test_maintenance_bonus_in_harvest(self) -> None:
        """r_harvest includes per-step maintenance bonus."""
        pref = _make_pref(3)
        # Pre-populate some mined ores to have nonzero potential
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        mined[3] = 10.0  # 10 diamonds already mined

        pot_before = _harvest_potential(
            mined, _REF_TOTAL, pref,
            _CFG.harvest_kappa, _CFG.harvest_epsilon,
        )
        assert pot_before > 0.0

        # Take a no-op step (no mining)
        r_harvest, _, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined,
            prev_potential=pot_before,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        # Delta is 0, but maintenance bonus > 0
        expected = _CFG.potential_maintenance_bonus * pot_before
        assert r_harvest == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Adjacent Ore Penalty
# ---------------------------------------------------------------------------


class TestAdjacentPenalty:
    def test_penalty_when_adjacent_desired_ore(self) -> None:
        """Penalty fires when adjacent to desired ore, not mining."""
        pref = _make_pref(3)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_adj, _, _, _, _ = compute_reward_components(
            action=4,  # TURN_LEFT
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_adj, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_adj, _, _, _, _ = compute_reward_components(
            action=6,  # DIG
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=1.0,  # 1 adjacent diamond
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        # mined_weight=1.0, adj=1.0 → raw_miss=0 → no penalty
        assert r_adj == 0.0

    def test_tanh_saturates_in_dense_veins(self) -> None:
        """Penalty is bounded even with many adjacent ores."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        # 5 adjacent coal ores, each w=1.0 → sum=5.0
        _, r_adj_5, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_skip0, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined.copy(),
            prev_potential=0.0,
            adjacent_desired_weight=1.0,
            adjacent_desired_weight_post=1.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=5,
        )
        assert r_skip5 < r_skip0  # more negative

    def test_skip_count_caps_at_max(self) -> None:
        """Skip count never exceeds adjacent_skip_cap."""
        pref = _make_pref(3)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        cap = _CFG.adjacent_skip_cap

        _, _, _, _, _, new_skip = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=1.0,
            adjacent_desired_weight_post=1.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=cap,
        )
        assert new_skip == cap  # stays at cap, does not exceed

    def test_skip_count_resets_on_mine(self) -> None:
        """Skip count resets to 0 when mining a desired ore."""
        pref = _make_pref(3)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, _, _, new_skip = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, _, _, new_skip = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        # adjacent_desired_weight should be 0 if only coal is adjacent
        _, r_adj, _, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,  # was already empty
            consecutive_skip_count=0,
        )
        assert r_clear == 0.0


# ---------------------------------------------------------------------------
# Operational Costs (Progressive Fuel Curve)
# ---------------------------------------------------------------------------


class TestOperationalCosts:
    def test_time_penalty_always_present(self) -> None:
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, r_ops, _, _ = compute_reward_components(
            action=4,  # TURN_LEFT
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r_ops == pytest.approx(_CFG.time_penalty)

    def test_progressive_fuel_penalty(self) -> None:
        """Fuel penalty ramps quadratically below threshold."""
        pref = _make_pref(0)
        max_fuel = 500
        threshold = _CFG.fuel_critical_threshold  # 0.2

        # At 20% fuel (boundary): no penalty
        fuel_at_boundary = int(threshold * max_fuel)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        _, _, _, r_ops_20, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=fuel_at_boundary),
            max_fuel=max_fuel,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        assert r_ops_20 == pytest.approx(_CFG.time_penalty)

        # At 10% fuel: progress=0.5, penalty = -1.0 * 0.25
        fuel_at_10 = int(0.1 * max_fuel)
        _, _, _, r_ops_10, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=fuel_at_10),
            max_fuel=max_fuel,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined.copy(),
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        expected_10 = _CFG.fuel_critical_penalty * 0.25 + _CFG.time_penalty
        assert r_ops_10 == pytest.approx(expected_10, abs=0.01)

        # At 0% fuel: full penalty
        _, _, _, r_ops_0, _, _ = compute_reward_components(
            action=0,
            block_mined=None,
            turtle=_FakeTurtle(fuel=0),
            max_fuel=max_fuel,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined.copy(),
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )
        expected_0 = _CFG.fuel_critical_penalty + _CFG.time_penalty
        assert r_ops_0 == pytest.approx(expected_0)

    def test_no_movement_or_dig_cost(self) -> None:
        """Movement and dig costs no longer exist."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        # FORWARD action — should only have time_penalty
        _, _, _, r_ops_move, _, _ = compute_reward_components(
            action=0,  # FORWARD
            block_mined=None,
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
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
            reference_total=_REF_TOTAL,
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
        expected = (
            _CFG.harvest_alpha * 0.01
            + (-0.05) + 0.2 + (-0.001)
        )
        assert r == pytest.approx(expected)

    def test_episode_bonus_deprecated(self) -> None:
        """Episode bonus is deprecated and returns 0."""
        bonus = PreferenceManager.compute_episode_bonus(0.7)
        assert bonus == 0.0


# ---------------------------------------------------------------------------
# Reward Range Bounds
# ---------------------------------------------------------------------------


class TestRewardBounds:
    def test_reward_range_bounded(self) -> None:
        """All per-step rewards stay within [-3, +3]."""
        rng = np.random.default_rng(42)
        pref = _make_pref(0)

        for _ in range(100):
            mined = rng.random(NUM_ORE_TYPES) * 50
            mined = mined.astype(np.float64)
            fuel = int(rng.integers(0, 500))
            adj_w = float(rng.random() * 3)
            skip = int(rng.integers(0, 20))

            action = int(rng.integers(0, 9))
            block = (
                int(BlockType.COAL_ORE)
                if action >= 6 else None
            )

            r_h, r_a, r_c, r_o, _, _ = compute_reward_components(
                action=action,
                block_mined=block,
                turtle=_FakeTurtle(fuel=fuel),
                max_fuel=500,
                preference=pref,
                reference_total=_REF_TOTAL,
                mined_ore_counts=mined.copy(),
                prev_potential=float(rng.random()),
                adjacent_desired_weight=adj_w,
                adjacent_desired_weight_post=adj_w * 0.5,
                prev_adjacent_weight=adj_w,
                consecutive_skip_count=skip,
            )
            total = PreferenceManager.scalarize(
                r_h, r_a, r_c, r_o,
            )
            assert -3.0 <= total <= 3.0, (
                f"Reward {total} out of bounds: "
                f"h={r_h}, a={r_a}, c={r_c}, o={r_o}"
            )

    def test_reference_total_deterministic(self) -> None:
        """Same mining on different reference totals gives same result
        when using a fixed reference."""
        pref = _make_pref(3)

        mined1 = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        r1, _, _, _, _, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined1,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )

        mined2 = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        r2, _, _, _, _, _ = compute_reward_components(
            action=6,
            block_mined=int(BlockType.DIAMOND_ORE),
            turtle=_FakeTurtle(fuel=100),
            max_fuel=500,
            preference=pref,
            reference_total=_REF_TOTAL,
            mined_ore_counts=mined2,
            prev_potential=0.0,
            adjacent_desired_weight=0.0,
            adjacent_desired_weight_post=0.0,
            prev_adjacent_weight=0.0,
            consecutive_skip_count=0,
        )

        assert r1 == pytest.approx(r2)


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


# ---------------------------------------------------------------------------
# Stage 1 Reward System
# ---------------------------------------------------------------------------


_S1_CFG = Stage1RewardConfig()


class TestStage1Reward:
    def test_target_ore_gives_positive_reward(self) -> None:
        """Mining a target ore gives per_ore_reward * preference."""
        pref = _make_pref(3)  # diamond
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        r_h, _, _, _, _ = compute_stage1_reward_components(
            block_mined=int(BlockType.DIAMOND_ORE),
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
        )
        # Default total_target_ores_in_world=100, reference=100 → mult=1.0
        assert r_h == pytest.approx(
            _S1_CFG.per_ore_reward * 1.0,
        )

    def test_no_harvest_for_non_target_ore(self) -> None:
        """Mining non-preferred ore gives zero harvest reward."""
        pref = _make_pref(3)  # diamond
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        r_h, _, _, _, _ = compute_stage1_reward_components(
            block_mined=int(BlockType.COAL_ORE),
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
        )
        assert r_h == 0.0

    def test_non_target_ore_increases_waste(self) -> None:
        """Mining non-preferred ore increases waste by multiplier."""
        pref = _make_pref(3)  # diamond
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, _, new_waste = compute_stage1_reward_components(
            block_mined=int(BlockType.COAL_ORE),
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
        )
        assert new_waste == int(_S1_CFG.non_target_ore_multiplier)

    def test_stone_mining_increases_waste(self) -> None:
        """Mining stone increases waste count by 1."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, _, new_waste = compute_stage1_reward_components(
            block_mined=int(BlockType.STONE),
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=5,
            is_new_position=False,
        )
        assert new_waste == 6

    def test_waste_penalty_soft_then_sharp(self) -> None:
        """Waste penalty starts near zero and ramps quadratically."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        # Low waste: very small penalty
        _, _, _, r_ops_low, _ = compute_stage1_reward_components(
            block_mined=int(BlockType.STONE),
            preference=pref,
            mined_ore_counts=mined.copy(),
            cumulative_waste_count=5,
            is_new_position=False,
        )

        # High waste: larger penalty
        _, _, _, r_ops_high, _ = compute_stage1_reward_components(
            block_mined=int(BlockType.STONE),
            preference=pref,
            mined_ore_counts=mined.copy(),
            cumulative_waste_count=80,
            is_new_position=False,
        )

        assert r_ops_low < 0.0
        assert r_ops_high < r_ops_low  # more negative

    def test_waste_penalty_caps_at_beta(self) -> None:
        """Waste penalty never exceeds -waste_beta per step."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, r_ops, _ = compute_stage1_reward_components(
            block_mined=int(BlockType.STONE),
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=500,  # way over ramp
            is_new_position=False,
        )
        # r_ops = time_penalty + waste_penalty
        assert r_ops >= _S1_CFG.time_penalty - _S1_CFG.waste_beta

    def test_exploration_bonus_on_new_cell(self) -> None:
        """New position gives exploration bonus."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=True,
        )
        assert r_clear == _S1_CFG.exploration_bonus

    def test_no_exploration_bonus_on_revisit(self) -> None:
        """Revisiting a cell gives zero exploration bonus."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
        )
        assert r_clear == 0.0

    def test_no_adjacent_penalty(self) -> None:
        """Stage 1 always returns r_adjacent=0.0."""
        pref = _make_pref(3)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_adj, _, _, _ = compute_stage1_reward_components(
            block_mined=int(BlockType.DIAMOND_ORE),
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=True,
        )
        assert r_adj == 0.0

    def test_mined_counts_mutated_in_place(self) -> None:
        """mined_ore_counts is updated in-place."""
        pref = _make_pref(3)  # diamond = index 3
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        compute_stage1_reward_components(
            block_mined=int(BlockType.DIAMOND_ORE),
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
        )
        assert mined[3] == 1.0

    def test_time_penalty_when_no_dig(self) -> None:
        """Time penalty applies even when no block is mined."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, r_ops, new_waste = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=50,
            is_new_position=False,
        )
        assert r_ops == pytest.approx(_S1_CFG.time_penalty)
        assert new_waste == 50  # unchanged


    def test_time_penalty_stacks_with_waste(self) -> None:
        """Time penalty is added on top of waste penalty."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, r_ops, _ = compute_stage1_reward_components(
            block_mined=int(BlockType.STONE),
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
        )
        # Should be time_penalty + waste penalty (both negative)
        assert r_ops < _S1_CFG.time_penalty

    def test_time_penalty_on_target_ore_step(self) -> None:
        """r_ops includes time penalty even on target ore steps."""
        pref = _make_pref(3)  # diamond
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, _, r_ops, _ = compute_stage1_reward_components(
            block_mined=int(BlockType.DIAMOND_ORE),
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
        )
        # Target ore has no waste penalty, only time penalty
        assert r_ops == pytest.approx(_S1_CFG.time_penalty)


class TestStage1TerminalBonus:
    def test_full_completion(self) -> None:
        """100% completion gives completion_scale."""
        pref = _make_pref(0)  # coal
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        mined[0] = 100.0

        bonus = compute_stage1_terminal_bonus(
            mined_ore_counts=mined,
            preference=pref,
            total_target_ores_in_world=100,
        )
        assert bonus == pytest.approx(_S1_CFG.completion_scale)

    def test_zero_completion(self) -> None:
        """0% completion gives zero bonus."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        bonus = compute_stage1_terminal_bonus(
            mined_ore_counts=mined,
            preference=pref,
            total_target_ores_in_world=100,
        )
        assert bonus == 0.0

    def test_partial_completion(self) -> None:
        """50% completion gives half the scale."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        mined[0] = 50.0

        bonus = compute_stage1_terminal_bonus(
            mined_ore_counts=mined,
            preference=pref,
            total_target_ores_in_world=100,
        )
        assert bonus == pytest.approx(
            _S1_CFG.completion_scale * 0.5,
        )

    def test_zero_target_ores_safe(self) -> None:
        """No target ores in world returns 0 bonus."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        mined[0] = 10.0

        bonus = compute_stage1_terminal_bonus(
            mined_ore_counts=mined,
            preference=pref,
            total_target_ores_in_world=0,
        )
        assert bonus == 0.0

    def test_completion_capped_at_one(self) -> None:
        """Completion ratio caps at 1.0 even if over-mined."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        mined[0] = 200.0

        bonus = compute_stage1_terminal_bonus(
            mined_ore_counts=mined,
            preference=pref,
            total_target_ores_in_world=100,
        )
        assert bonus == pytest.approx(_S1_CFG.completion_scale)


class TestStage1XZExploration:
    """Tests for XZ-plane exploration bonus in Stage 1."""

    def test_xz_bonus_at_correct_depth(self) -> None:
        """New XZ cell at correct Y-depth gives XZ exploration bonus."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            turtle_y=10,
            ore_y_range=(5.0, 15.0),
            is_new_xz_position=True,
            explored_xz_count=0,
        )
        assert r_clear == pytest.approx(_S1_CFG.xz_exploration_bonus)

    def test_no_xz_bonus_outside_y_range(self) -> None:
        """New XZ cell outside Y-range gives no XZ bonus."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            turtle_y=25,
            ore_y_range=(5.0, 15.0),
            is_new_xz_position=True,
            explored_xz_count=0,
            prev_y_dist=10.0,  # match current y_dist so progress = 0
        )
        assert r_clear == 0.0

    def test_no_xz_bonus_on_revisit(self) -> None:
        """Revisiting an XZ column gives no XZ bonus."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            turtle_y=10,
            ore_y_range=(5.0, 15.0),
            is_new_xz_position=False,
            explored_xz_count=50,
        )
        assert r_clear == 0.0

    def test_xz_bonus_decays_with_count(self) -> None:
        """XZ bonus decays as more XZ cells are explored."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        base_kwargs = dict(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            turtle_y=10,
            ore_y_range=(5.0, 15.0),
            is_new_xz_position=True,
        )

        _, _, r_early, _, _ = compute_stage1_reward_components(
            **base_kwargs, explored_xz_count=0,
        )
        _, _, r_late, _, _ = compute_stage1_reward_components(
            **base_kwargs, explored_xz_count=200,
        )
        assert r_early > r_late > 0.0

    def test_xz_bonus_stacks_with_3d_exploration(self) -> None:
        """XZ bonus and 3D exploration bonus can fire together."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=True,
            turtle_y=10,
            ore_y_range=(5.0, 15.0),
            is_new_xz_position=True,
            explored_xz_count=0,
            explored_count=0,
        )
        expected = (
            _S1_CFG.exploration_bonus
            + _S1_CFG.xz_exploration_bonus
        )
        assert r_clear == pytest.approx(expected)

    def test_xz_bonus_at_y_range_boundary(self) -> None:
        """XZ bonus fires exactly at the Y-range boundaries."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        base_kwargs = dict(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            ore_y_range=(5.0, 15.0),
            is_new_xz_position=True,
            explored_xz_count=0,
        )

        # At y_min boundary
        _, _, r_at_min, _, _ = compute_stage1_reward_components(
            **base_kwargs, turtle_y=5,
        )
        assert r_at_min == pytest.approx(_S1_CFG.xz_exploration_bonus)

        # At y_max boundary
        _, _, r_at_max, _, _ = compute_stage1_reward_components(
            **base_kwargs, turtle_y=15,
        )
        assert r_at_max == pytest.approx(_S1_CFG.xz_exploration_bonus)

        # One above y_max — no bonus (prev_y_dist=1 matches current y_dist)
        _, _, r_above, _, _ = compute_stage1_reward_components(
            **base_kwargs, turtle_y=16, prev_y_dist=1.0,
        )
        assert r_above == 0.0


class TestStage1QuadraticYPenalty:
    """Tests for quadratic Y-distance penalty in Stage 1."""

    def test_quadratic_scaling(self) -> None:
        """Y-penalty uses quadratic scaling with base cost."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_adj, _, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            turtle_y=25,
            ore_y_range=(5.0, 15.0),
            world_height=40,
            prev_y_dist=10.0,
        )
        # y_dist=10, frac=10/40=0.25, penalty = -(1.0*0.0625) - 0.05
        expected = -_S1_CFG.y_penalty_scale * 0.25 ** 2 - _S1_CFG.y_penalty_base
        assert r_adj == pytest.approx(expected)

    def test_penalty_steeper_at_large_distance(self) -> None:
        """Quadratic penalty grows faster for large distances."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        base_kwargs = dict(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            ore_y_range=(0.0, 8.0),
            world_height=40,
        )

        _, r_near, _, _, _ = compute_stage1_reward_components(
            **base_kwargs, turtle_y=12, prev_y_dist=4.0,
        )
        _, r_far, _, _, _ = compute_stage1_reward_components(
            **base_kwargs, turtle_y=28, prev_y_dist=20.0,
        )
        # Both should be negative, far should be worse
        assert r_near < 0
        assert r_far < r_near

    def test_no_penalty_in_range(self) -> None:
        """No Y-penalty when at correct depth."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, r_adj, _, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            turtle_y=10,
            ore_y_range=(5.0, 15.0),
        )
        assert r_adj == _S1_CFG.y_in_range_bonus


class TestStage1VerticalProgress:
    """Tests for vertical progress shaping in Stage 1."""

    def test_positive_reward_for_closing_distance(self) -> None:
        """Moving closer to target Y-range gives positive r_clear."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            turtle_y=20,
            ore_y_range=(5.0, 15.0),
            world_height=40,
            prev_y_dist=6.0,  # was 6 blocks away, now 5
        )
        # y_dist=5, progress = 1.0*(6/40 - 5/40) = 0.025
        assert r_clear == pytest.approx(
            _S1_CFG.y_progress_scale * (6.0 / 40 - 5.0 / 40),
        )

    def test_negative_reward_for_increasing_distance(self) -> None:
        """Moving away from target Y-range gives negative r_clear."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            turtle_y=22,
            ore_y_range=(5.0, 15.0),
            world_height=40,
            prev_y_dist=6.0,  # was 6 blocks away, now 7
        )
        # y_dist=7, progress = 1.0*(6/40 - 7/40) = -0.025
        assert r_clear < 0

    def test_zero_progress_when_stationary(self) -> None:
        """No progress reward when Y-distance unchanged."""
        pref = _make_pref(0)
        mined = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        _, _, r_clear, _, _ = compute_stage1_reward_components(
            block_mined=None,
            preference=pref,
            mined_ore_counts=mined,
            cumulative_waste_count=0,
            is_new_position=False,
            turtle_y=20,
            ore_y_range=(5.0, 15.0),
            world_height=40,
            prev_y_dist=5.0,  # same as current y_dist
        )
        assert r_clear == pytest.approx(0.0)
