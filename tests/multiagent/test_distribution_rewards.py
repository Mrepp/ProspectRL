"""Unit tests for SpawnRateNormalizer and distribution-aware rewards.

Validates:
- Rarity multipliers derived from worldgen data
- MINE_ORE reward weighting by preference * rarity
- Tolerant treatment of incidental (non-preferred) ore
- Potential-based alignment bonus
- Per-agent preference blending
- TaskType.MINE_ORE enum presence
"""

from __future__ import annotations

import numpy as np
import pytest

from prospect_rl.config import NUM_ORE_TYPES, ORE_INDEX, ORE_TYPES
from prospect_rl.multiagent.agent.task_rewards import (
    SpawnRateNormalizer,
    _compute_alignment,
    _set_normalizer,
    compute_task_reward,
)
from prospect_rl.multiagent.coordinator.assignment import (
    BoundingBox,
    TaskAssignment,
    TaskType,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_bbox() -> BoundingBox:
    return BoundingBox(
        x_min=0, x_max=63, z_min=0, z_max=63,
        y_min=0, y_max=63,
    )


def _make_mine_ore_assignment(
    preference: np.ndarray | None = None,
) -> TaskAssignment:
    if preference is None:
        preference = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        preference[3] = 1.0  # diamond
    return TaskAssignment(
        agent_id=0,
        task_type=TaskType.MINE_ORE,
        target_position=np.array([32, 32, 32], dtype=np.int32),
        bounding_box=_make_bbox(),
        ore_preference=preference,
        seed_position=np.array([32, 32, 32], dtype=np.int32),
    )


def _ore_block_type(ore_name: str) -> int:
    """Get block type int for a named ore (e.g. 'diamond')."""
    for bt in ORE_TYPES:
        if ore_name in bt.name.lower():
            return int(bt)
    raise ValueError(f"Unknown ore: {ore_name}")


# ── Test 1: Rarity from worldgen ─────────────────────────────────────


class TestRarityFromWorldgen:
    """Verify SpawnRateNormalizer loads from worldgen parser."""

    def test_rarity_ordering(self) -> None:
        """Diamond should have higher rarity mult than iron > coal."""
        normalizer = SpawnRateNormalizer(rarity_mult_cap=10.0)
        rm = normalizer.rarity_mult

        # Find indices
        coal_idx = None
        iron_idx = None
        diamond_idx = None
        for i, bt in enumerate(ORE_TYPES):
            name = bt.name.lower()
            if "coal" in name:
                coal_idx = i
            elif "iron" in name:
                iron_idx = i
            elif "diamond" in name:
                diamond_idx = i

        assert coal_idx is not None
        assert iron_idx is not None
        assert diamond_idx is not None

        # Diamond rarer than iron, iron rarer than coal
        assert rm[diamond_idx] > rm[iron_idx], (
            f"diamond rarity {rm[diamond_idx]} should > "
            f"iron {rm[iron_idx]}"
        )
        assert rm[iron_idx] >= rm[coal_idx], (
            f"iron rarity {rm[iron_idx]} should >= "
            f"coal {rm[coal_idx]}"
        )

    def test_expected_per_chunk_positive(self) -> None:
        """All tracked ores should have positive expected blocks."""
        normalizer = SpawnRateNormalizer()
        epc = normalizer.expected_per_chunk
        # At least coal, iron, diamond should be positive
        assert epc.sum() > 0

    def test_cap_respected(self) -> None:
        """Rarity mult should not exceed cap."""
        cap = 3.0
        normalizer = SpawnRateNormalizer(rarity_mult_cap=cap)
        rm = normalizer.rarity_mult
        assert np.all(rm <= cap + 1e-6)


# ── Test 2: MINE_ORE prefers target ──────────────────────────────────


class TestMineOrePreferTarget:
    """Mining preferred ore should give higher reward than non-pref."""

    def test_diamond_pref_rewards_diamond_more(self) -> None:
        # Set up a normalizer with uniform rarity for clarity
        normalizer = SpawnRateNormalizer()
        _set_normalizer(normalizer)

        pref = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        pref[3] = 1.0  # diamond
        assignment = _make_mine_ore_assignment(pref)

        pos = np.array([32, 32, 32], dtype=np.float32)
        prev_pos = pos.copy()
        counts = np.zeros(NUM_ORE_TYPES, dtype=np.int32)

        diamond_bt = _ore_block_type("diamond")
        coal_bt = _ore_block_type("coal")

        # Mine diamond
        r_diamond, _, _ = compute_task_reward(
            assignment=assignment,
            agent_position=pos,
            prev_position=prev_pos,
            block_mined=diamond_bt,
            blocks_cleared=1,
            step_budget=0,
            task_steps=1,
            world_diagonal=100.0,
            mined_ore_counts=counts,
        )

        # Mine coal (non-preferred)
        r_coal, _, _ = compute_task_reward(
            assignment=assignment,
            agent_position=pos,
            prev_position=prev_pos,
            block_mined=coal_bt,
            blocks_cleared=1,
            step_budget=0,
            task_steps=1,
            world_diagonal=100.0,
            mined_ore_counts=counts,
        )

        assert r_diamond > r_coal, (
            f"Diamond reward {r_diamond} should > coal {r_coal}"
        )


# ── Test 3: Tolerant of incidental mining ─────────────────────────────


class TestTolerantOfIncidental:
    """Mining non-preferred ore should give positive reward."""

    def test_non_preferred_ore_positive(self) -> None:
        normalizer = SpawnRateNormalizer()
        _set_normalizer(normalizer)

        pref = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        pref[3] = 1.0  # diamond only
        assignment = _make_mine_ore_assignment(pref)

        pos = np.array([32, 32, 32], dtype=np.float32)
        coal_bt = _ore_block_type("coal")
        counts = np.zeros(NUM_ORE_TYPES, dtype=np.int32)

        r, _, _ = compute_task_reward(
            assignment=assignment,
            agent_position=pos,
            prev_position=pos.copy(),
            block_mined=coal_bt,
            blocks_cleared=1,
            step_budget=0,
            task_steps=1,
            world_diagonal=100.0,
            mined_ore_counts=counts,
        )

        # Should be positive (tolerant), not negative
        # Boundary bonus (box_stay_bonus=0.05) + progress_reward*0.3
        assert r > 0, (
            f"Non-preferred ore reward {r} should be positive"
        )


# ── Test 4: Alignment is potential-based ──────────────────────────────


class TestAlignmentPotentialBased:
    """Mining under-represented preferred ore gives positive delta."""

    def test_alignment_improves_when_mining_target(self) -> None:
        pref = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        pref[3] = 0.6  # diamond
        pref[0] = 0.4  # coal

        rarity = np.ones(NUM_ORE_TYPES, dtype=np.float32)

        # Start: 5 coal mined, 0 diamond (under-represented)
        counts_before = np.zeros(NUM_ORE_TYPES, dtype=np.int32)
        counts_before[0] = 5

        align_before = _compute_alignment(
            counts_before, pref, rarity,
        )

        # Mine a diamond
        counts_after = counts_before.copy()
        counts_after[3] = 1

        align_after = _compute_alignment(
            counts_after, pref, rarity,
        )

        # Alignment should improve
        assert align_after > align_before, (
            f"Alignment should improve: {align_before} -> "
            f"{align_after}"
        )

    def test_alignment_zero_when_no_mining(self) -> None:
        pref = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        pref[3] = 1.0
        rarity = np.ones(NUM_ORE_TYPES, dtype=np.float32)
        counts = np.zeros(NUM_ORE_TYPES, dtype=np.int32)
        assert _compute_alignment(counts, pref, rarity) == 0.0


# ── Test 5: Per-agent preference differs ──────────────────────────────


class TestPerAgentPreferenceDiffers:
    """Different chunks produce different per-agent preferences."""

    def test_different_chunks_different_prefs(self) -> None:
        from prospect_rl.multiagent.belief_map import ChunkState

        team_pref = np.array(
            [0.3, 0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        # Chunk A: lots of coal (idx 0)
        chunk_a = ChunkState(
            chunk_coords=(0, 0), total_voxels=4096,
        )
        chunk_a.expected_remaining = np.zeros(
            NUM_ORE_TYPES, dtype=np.float32,
        )
        chunk_a.expected_remaining[0] = 100.0  # coal-rich

        # Chunk B: lots of diamond (idx 3)
        chunk_b = ChunkState(
            chunk_coords=(1, 0), total_voxels=4096,
        )
        chunk_b.expected_remaining = np.zeros(
            NUM_ORE_TYPES, dtype=np.float32,
        )
        chunk_b.expected_remaining[3] = 50.0  # diamond-rich

        # Use a mock with _preference_blend_alpha = 0.5
        class _MockCoord:
            _preference_blend_alpha = 0.5

        from prospect_rl.multiagent.coordinator.coordinator import (
            Coordinator,
        )

        mock = _MockCoord()
        pref_a = Coordinator._compute_agent_preference(
            mock, chunk_a, team_pref,
        )
        pref_b = Coordinator._compute_agent_preference(
            mock, chunk_b, team_pref,
        )

        # Should differ
        assert not np.allclose(pref_a, pref_b), (
            f"Per-agent prefs should differ: {pref_a} vs {pref_b}"
        )

        # Both should sum to ~1.0
        assert abs(float(np.sum(pref_a)) - 1.0) < 1e-5
        assert abs(float(np.sum(pref_b)) - 1.0) < 1e-5


# ── Test 6: Rename verified ───────────────────────────────────────────


class TestRenameMineOre:
    """Verify TaskType.MINE_ORE exists and MINE_VEIN does not."""

    def test_mine_ore_exists(self) -> None:
        assert hasattr(TaskType, "MINE_ORE")
        assert TaskType.MINE_ORE == 2

    def test_mine_vein_absent(self) -> None:
        assert not hasattr(TaskType, "MINE_VEIN")
