"""Tests for Phase 1 pre-training fixes.

Validates:
- Fix 1: MOVE_TO progress reward is flat per-block (~0.1), not diagonal-normalized (~0.006)
- Fix 2: Corridor bbox contains both agent start and target end
- Fix 2: MINE_ORE bbox centered on agent position
- Fix 2: RETURN_TO corridor bbox contains agent start
"""

from __future__ import annotations

import numpy as np
import pytest

from prospect_rl.config import NUM_ORE_TYPES
from prospect_rl.multiagent.agent.task_rewards import compute_task_reward
from prospect_rl.multiagent.coordinator.assignment import (
    BoundingBox,
    TaskAssignment,
    TaskType,
)
from prospect_rl.multiagent.training.task_sampler import TaskSampler


# ── Helpers ──────────────────────────────────────────────────────────


def _make_bbox() -> BoundingBox:
    return BoundingBox(
        x_min=0, x_max=63, z_min=0, z_max=63,
        y_min=0, y_max=127,
    )


def _one_hot_pref(idx: int = 0) -> np.ndarray:
    pref = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
    pref[idx] = 1.0
    return pref


def _make_move_to(
    target: np.ndarray,
    bbox: BoundingBox | None = None,
) -> TaskAssignment:
    if bbox is None:
        bbox = _make_bbox()
    return TaskAssignment(
        agent_id=0,
        task_type=TaskType.MOVE_TO,
        target_position=target,
        bounding_box=bbox,
        ore_preference=_one_hot_pref(),
    )


# ── Fix 1: MOVE_TO progress reward magnitude ────────────────────────


class TestMoveToProgressReward:
    """Progress reward should be ~0.1 per block, not ~0.006."""

    def test_one_block_closer_gives_progress_reward(self) -> None:
        """Moving 1 block closer should yield >= progress_reward (0.1)."""
        target = np.array([50, 50, 50], dtype=np.int32)
        agent_pos = np.array([40, 50, 50], dtype=np.float32)  # 10 blocks away
        prev_pos = np.array([39, 50, 50], dtype=np.float32)  # 11 blocks away

        assignment = _make_move_to(target)

        r, complete, _ = compute_task_reward(
            assignment=assignment,
            agent_position=agent_pos,
            prev_position=prev_pos,
            block_mined=None,
            blocks_cleared=0,
            step_budget=0,
            task_steps=1,
            world_diagonal=157.0,  # typical for 64x128x64
        )

        # Progress reward should be at least 0.1 (the default progress_reward)
        # Before fix: 0.1 * (1/157) * 10 = 0.006
        assert r >= 0.08, (
            f"MOVE_TO progress for 1-block move should be >= 0.08, got {r}"
        )

    def test_regression_penalty(self) -> None:
        """Moving away should incur regress_penalty (-0.02)."""
        target = np.array([50, 50, 50], dtype=np.int32)
        agent_pos = np.array([39, 50, 50], dtype=np.float32)  # 11 away
        prev_pos = np.array([40, 50, 50], dtype=np.float32)  # 10 away

        assignment = _make_move_to(target)

        r, _, _ = compute_task_reward(
            assignment=assignment,
            agent_position=agent_pos,
            prev_position=prev_pos,
            block_mined=None,
            blocks_cleared=0,
            step_budget=0,
            task_steps=1,
            world_diagonal=157.0,
        )

        # Should include regress_penalty (-0.02)
        assert r <= -0.01, (
            f"MOVE_TO regression should be negative, got {r}"
        )

    def test_completion_fires(self) -> None:
        """Reaching within 1 block of target triggers completion."""
        target = np.array([50, 50, 50], dtype=np.int32)
        agent_pos = np.array([50, 50, 50], dtype=np.float32)  # at target

        assignment = _make_move_to(target)

        _, complete, _ = compute_task_reward(
            assignment=assignment,
            agent_position=agent_pos,
            prev_position=np.array([49, 50, 50], dtype=np.float32),
            block_mined=None,
            blocks_cleared=0,
            step_budget=0,
            task_steps=1,
            world_diagonal=157.0,
        )

        assert complete, "Should complete when at target"

    def test_reward_independent_of_world_diagonal(self) -> None:
        """Progress reward should be the same regardless of world size."""
        target = np.array([50, 50, 50], dtype=np.int32)
        agent_pos = np.array([40, 50, 50], dtype=np.float32)
        prev_pos = np.array([39, 50, 50], dtype=np.float32)

        assignment = _make_move_to(target)

        r_small, _, _ = compute_task_reward(
            assignment=assignment,
            agent_position=agent_pos,
            prev_position=prev_pos,
            block_mined=None,
            blocks_cleared=0,
            step_budget=0,
            task_steps=1,
            world_diagonal=50.0,  # small world
        )

        r_large, _, _ = compute_task_reward(
            assignment=assignment,
            agent_position=agent_pos,
            prev_position=prev_pos,
            block_mined=None,
            blocks_cleared=0,
            step_budget=0,
            task_steps=1,
            world_diagonal=500.0,  # large world
        )

        assert abs(r_small - r_large) < 0.01, (
            f"Reward should not depend on world_diagonal: "
            f"small={r_small}, large={r_large}"
        )


# ── Fix 2: Corridor bbox containment ────────────────────────────────


class TestCorridorBbox:
    """Corridor bbox should contain both agent start and target end."""

    def _make_sampler(
        self, world_size: tuple[int, int, int] = (64, 128, 64),
    ) -> TaskSampler:
        return TaskSampler(world_size=world_size, seed=42)

    def test_move_to_bbox_contains_agent(self) -> None:
        """MOVE_TO bbox should contain agent's starting position."""
        sampler = self._make_sampler()
        agent_pos = np.array([10, 10, 10], dtype=np.int32)

        task = sampler.sample(
            agent_id=0,
            agent_position=agent_pos,
            preference=_one_hot_pref(),
        )

        # Force a MOVE_TO by sampling repeatedly (or test the method directly)
        # Use internal method for determinism
        task = sampler._sample_move_to(0, agent_pos, _one_hot_pref())
        bbox = task.bounding_box

        assert bbox.contains(
            int(agent_pos[0]), int(agent_pos[1]), int(agent_pos[2]),
        ), (
            f"Agent at {agent_pos} should be inside MOVE_TO bbox "
            f"[{bbox.x_min}:{bbox.x_max}, {bbox.y_min}:{bbox.y_max}, "
            f"{bbox.z_min}:{bbox.z_max}]"
        )

    def test_move_to_bbox_contains_target(self) -> None:
        """MOVE_TO bbox should contain the target position."""
        sampler = self._make_sampler()
        agent_pos = np.array([10, 10, 10], dtype=np.int32)

        task = sampler._sample_move_to(0, agent_pos, _one_hot_pref())
        bbox = task.bounding_box
        target = task.target_position

        assert bbox.contains(
            int(target[0]), int(target[1]), int(target[2]),
        ), (
            f"Target at {target} should be inside MOVE_TO bbox "
            f"[{bbox.x_min}:{bbox.x_max}, {bbox.y_min}:{bbox.y_max}, "
            f"{bbox.z_min}:{bbox.z_max}]"
        )

    def test_move_to_far_agent_still_inside(self) -> None:
        """Agent far from target should still be inside corridor bbox."""
        sampler = self._make_sampler()
        sampler.set_difficulty(2)  # Hard: max_dist=40

        agent_pos = np.array([5, 5, 5], dtype=np.int32)
        task = sampler._sample_move_to(0, agent_pos, _one_hot_pref())
        bbox = task.bounding_box

        assert bbox.contains(5, 5, 5), (
            f"Agent at corner should be inside corridor bbox"
        )

    def test_return_to_bbox_contains_agent(self) -> None:
        """RETURN_TO bbox should contain agent's starting position."""
        sampler = self._make_sampler()
        agent_pos = np.array([5, 5, 5], dtype=np.int32)

        task = sampler._sample_return_to(0, agent_pos, _one_hot_pref())
        bbox = task.bounding_box

        assert bbox.contains(5, 5, 5), (
            f"Agent at {agent_pos} should be inside RETURN_TO bbox"
        )

    def test_return_to_bbox_contains_base(self) -> None:
        """RETURN_TO bbox should contain the base (world center)."""
        sampler = self._make_sampler()
        agent_pos = np.array([5, 5, 5], dtype=np.int32)

        task = sampler._sample_return_to(0, agent_pos, _one_hot_pref())
        bbox = task.bounding_box
        target = task.target_position

        assert bbox.contains(
            int(target[0]), int(target[1]), int(target[2]),
        ), (
            f"Base at {target} should be inside RETURN_TO bbox"
        )


# ── Fix 2: MINE_ORE bbox centered on agent ──────────────────────────


class TestMineOreBbox:
    """MINE_ORE bbox should be centered on agent, not random position."""

    def test_mine_ore_bbox_contains_agent(self) -> None:
        """Agent should always start inside MINE_ORE bbox."""
        sampler = TaskSampler(world_size=(64, 128, 64), seed=42)
        agent_pos = np.array([20, 30, 20], dtype=np.int32)

        task = sampler._sample_mine_ore(0, agent_pos, _one_hot_pref())
        bbox = task.bounding_box

        assert bbox.contains(20, 30, 20), (
            f"Agent at {agent_pos} should be inside MINE_ORE bbox "
            f"[{bbox.x_min}:{bbox.x_max}, {bbox.y_min}:{bbox.y_max}, "
            f"{bbox.z_min}:{bbox.z_max}]"
        )

    def test_mine_ore_seed_pos_matches_agent(self) -> None:
        """MINE_ORE seed position should match agent position."""
        sampler = TaskSampler(world_size=(64, 128, 64), seed=42)
        agent_pos = np.array([20, 30, 20], dtype=np.int32)

        task = sampler._sample_mine_ore(0, agent_pos, _one_hot_pref())

        np.testing.assert_array_equal(
            task.target_position, agent_pos,
            err_msg="MINE_ORE seed should be at agent position",
        )

    def test_mine_ore_bbox_scales_with_difficulty(self) -> None:
        """MINE_ORE bbox should grow with difficulty."""
        sampler = TaskSampler(world_size=(64, 128, 64), seed=42)
        agent_pos = np.array([32, 64, 32], dtype=np.int32)

        sampler.set_difficulty(0)
        task_easy = sampler._sample_mine_ore(0, agent_pos, _one_hot_pref())
        size_easy = (
            task_easy.bounding_box.x_max - task_easy.bounding_box.x_min
        )

        sampler.set_difficulty(2)
        task_hard = sampler._sample_mine_ore(0, agent_pos, _one_hot_pref())
        size_hard = (
            task_hard.bounding_box.x_max - task_hard.bounding_box.x_min
        )

        assert size_hard > size_easy, (
            f"Hard difficulty bbox ({size_hard}) should be larger "
            f"than easy ({size_easy})"
        )


# ── Fix 2: Corridor bbox method unit test ────────────────────────────


class TestMakeCorridorBbox:
    """Direct test of _make_corridor_bbox method."""

    def test_corridor_contains_both_endpoints(self) -> None:
        sampler = TaskSampler(world_size=(64, 128, 64), seed=42)
        start = np.array([10, 10, 10], dtype=np.int32)
        end = np.array([50, 100, 50], dtype=np.int32)

        bbox = sampler._make_corridor_bbox(start, end, margin=4)

        assert bbox.contains(10, 10, 10), "Start should be inside"
        assert bbox.contains(50, 100, 50), "End should be inside"

    def test_corridor_respects_world_bounds(self) -> None:
        sampler = TaskSampler(world_size=(64, 128, 64), seed=42)
        start = np.array([0, 0, 0], dtype=np.int32)
        end = np.array([63, 127, 63], dtype=np.int32)

        bbox = sampler._make_corridor_bbox(start, end, margin=10)

        assert bbox.x_min >= 0
        assert bbox.x_max <= 63
        assert bbox.y_min >= 0
        assert bbox.y_max <= 127
        assert bbox.z_min >= 0
        assert bbox.z_max <= 63

    def test_corridor_includes_margin(self) -> None:
        sampler = TaskSampler(world_size=(64, 128, 64), seed=42)
        start = np.array([20, 20, 20], dtype=np.int32)
        end = np.array([40, 40, 40], dtype=np.int32)

        bbox = sampler._make_corridor_bbox(start, end, margin=4)

        assert bbox.x_min == 16  # 20 - 4
        assert bbox.x_max == 44  # 40 + 4
        assert bbox.y_min == 16
        assert bbox.y_max == 44
