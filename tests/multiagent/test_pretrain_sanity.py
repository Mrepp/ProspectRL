"""Pre-training sanity tests for multi-agent system.

Validates the 8 critical checks from the Phase 1 audit before
any training run.  Tests focus on data integrity, shape correctness,
and absence of silent corruption (NaN, occupancy drift, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prospect_rl.config import NUM_ORE_TYPES, BlockType
from prospect_rl.env.turtle import Turtle
from prospect_rl.env.world.world import World
from prospect_rl.multiagent.coordinator.gnn import (
    AGENT_FEATURE_DIM,
    CoordinatorGNNSimple,
    REGION_FEATURE_DIM,
)
from prospect_rl.multiagent.shared_world import SharedWorld
from prospect_rl.multiagent.telemetry import TelemetryEvent, TelemetryEventType
from prospect_rl.multiagent.training.coordinator_trainer import CoordinatorTrainer
from prospect_rl.multiagent.training.task_sampler import TaskSampler


# ── Helpers ──────────────────────────────────────────────────────────

def _make_small_world(
    size: tuple[int, int, int] = (32, 32, 32), seed: int = 42,
) -> World:
    """Create a small procedural world for testing."""
    return World(
        size=size, seed=seed, ore_density_multiplier=3.0,
        caves_enabled=False,
    )


def _make_shared_world(
    size: tuple[int, int, int] = (32, 32, 32),
    seed: int = 42,
    max_agents: int = 8,
) -> SharedWorld:
    world = _make_small_world(size=size, seed=seed)
    return SharedWorld(world=world, max_agents=max_agents)


# ── Test 1: Flush telemetry list independence (B5) ───────────────────

class TestFlushTelemetryIndependence:
    """Verify flush_telemetry() returns independent copies of lists."""

    def test_mutating_returned_list_does_not_affect_buffer(self) -> None:
        sw = _make_shared_world()
        events_1 = [
            TelemetryEvent(
                event_type=TelemetryEventType.BLOCK_OBSERVED,
                agent_id=0, position=(5, 5, 5),
                block_type=int(BlockType.STONE),
                previous_belief=0.0, timestamp=1,
            ),
        ]
        sw.record_telemetry(0, events_1)
        flushed = sw.flush_telemetry()

        # Mutate the returned list
        assert len(flushed[0]) == 1
        flushed[0].clear()

        # Record new events and flush again
        events_2 = [
            TelemetryEvent(
                event_type=TelemetryEventType.BLOCK_REMOVED,
                agent_id=0, position=(6, 6, 6),
                block_type=int(BlockType.COAL_ORE),
                previous_belief=0.0, timestamp=2,
            ),
        ]
        sw.record_telemetry(0, events_2)
        flushed_2 = sw.flush_telemetry()

        # Second flush must contain the new event, unaffected by the clear()
        assert len(flushed_2[0]) == 1
        assert flushed_2[0][0].event_type == TelemetryEventType.BLOCK_REMOVED


# ── Test 2: GNN forward pass shapes ─────────────────────────────────

class TestGNNForwardPassShapes:
    """Verify CoordinatorGNNSimple produces correct output shapes."""

    def test_output_shape_and_no_nan(self) -> None:
        gnn = CoordinatorGNNSimple()
        n_agents, n_regions = 4, 8
        x_dict = {
            "agent": torch.randn(n_agents, AGENT_FEATURE_DIM),
            "region": torch.randn(n_regions, REGION_FEATURE_DIM),
        }
        with torch.no_grad():
            scores = gnn(x_dict)
        assert scores.shape == (n_agents, n_regions)
        assert not torch.isnan(scores).any(), "NaN detected in GNN output"
        assert torch.isfinite(scores).all(), "Inf detected in GNN output"


# ── Test 3: Hungarian matching filters padded indices (B1) ───────────

class TestHungarianNoPaddedIndices:
    """Ensure _last_col_idx has no values >= n_regions after plan()."""

    def test_more_agents_than_regions(self) -> None:
        """8 agents with a small world that has few chunks."""
        size = (32, 32, 32)
        sw = _make_shared_world(size=size, seed=42, max_agents=10)

        # Register 8 agents at scattered positions
        agents = []
        for i in range(8):
            pos = np.array([4 + i * 3, 16, 16], dtype=np.int32)
            sw[pos[0], pos[1], pos[2]] = BlockType.AIR
            t = Turtle(position=pos)
            ok = sw.register_agent(i, t)
            if ok:
                agents.append(i)

        if len(agents) < 2:
            pytest.skip("Could not register enough agents")

        # Build minimal belief map + coordinator
        from prospect_rl.multiagent.belief_map import BeliefMap
        from prospect_rl.multiagent.coordinator.coordinator import Coordinator
        from prospect_rl.multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=size[1],
            apply_solid_correction=False,
            apply_calibration=False,
        )
        bm = BeliefMap(
            world_size=size,
            biome_map=sw._world.biome_map,
            prior=prior,
            chunk_size_xz=16,
        )
        pref = np.ones(NUM_ORE_TYPES, dtype=np.float32) / NUM_ORE_TYPES
        coord = Coordinator(
            belief_map=bm,
            shared_world=sw,
            preference=pref,
            chunk_size_xz=16,
        )

        coord.plan()

        # Verify stored indices are valid
        if coord._last_col_idx is not None and len(coord._last_col_idx) > 0:
            n_regions = len(bm._chunks)
            if n_regions > 0:
                assert np.all(coord._last_col_idx < n_regions), (
                    f"Padded indices found: max col={coord._last_col_idx.max()}, "
                    f"n_regions={n_regions}"
                )


# ── Test 4: REINFORCE with empty and valid assignments (B4) ─────────

class TestReinforceEmptyAndValid:
    """Ensure REINFORCE update handles empty and valid assignments."""

    def test_empty_assignment_no_nan(self) -> None:
        gnn = CoordinatorGNNSimple()
        trainer = CoordinatorTrainer(gnn=gnn)

        x_dict = {
            "agent": torch.randn(4, AGENT_FEATURE_DIM),
            "region": torch.randn(3, REGION_FEATURE_DIM),
        }

        # Empty assignment
        metrics = trainer.update(
            x_dict=x_dict,
            edge_index_dict={},
            row_idx=np.array([], dtype=np.int64),
            col_idx=np.array([], dtype=np.int64),
            team_reward=1.0,
        )
        assert metrics["coordinator/loss"] == 0.0

        # Verify no NaN in GNN parameters
        for name, p in gnn.named_parameters():
            assert torch.isfinite(p).all(), f"NaN/Inf in {name} after empty update"

    def test_valid_assignment_updates_weights(self) -> None:
        gnn = CoordinatorGNNSimple()
        trainer = CoordinatorTrainer(gnn=gnn)

        x_dict = {
            "agent": torch.randn(4, AGENT_FEATURE_DIM),
            "region": torch.randn(3, REGION_FEATURE_DIM),
        }

        # Snapshot weights before
        before = {n: p.data.clone() for n, p in gnn.named_parameters()}

        trainer.update(
            x_dict=x_dict,
            edge_index_dict={},
            row_idx=np.array([0, 1, 2], dtype=np.int64),
            col_idx=np.array([0, 1, 2], dtype=np.int64),
            team_reward=5.0,
        )

        # Verify no NaN
        for name, p in gnn.named_parameters():
            assert torch.isfinite(p).all(), f"NaN/Inf in {name} after valid update"

        # Verify some weights changed (training happened)
        any_changed = any(
            not torch.equal(before[n], p.data)
            for n, p in gnn.named_parameters()
        )
        assert any_changed, "GNN weights did not change after valid update"


# ── Test 5: Task sampler distribution (N2) ───────────────────────────

class TestTaskSamplerDistribution:
    """Verify task type sampling roughly matches configured weights."""

    def test_distribution_matches_weights(self) -> None:
        sampler = TaskSampler(world_size=(64, 64, 64), seed=42)

        from prospect_rl.multiagent.coordinator.assignment import TaskType

        counts = dict.fromkeys(TaskType, 0)
        n_samples = 2000
        pref = np.ones(NUM_ORE_TYPES, dtype=np.float32) / NUM_ORE_TYPES
        pos = np.array([32, 32, 32], dtype=np.int32)

        for _ in range(n_samples):
            task = sampler.sample(
                agent_id=0,
                agent_position=pos,
                preference=pref,
            )
            counts[task.task_type] += 1

        # Expected: MOVE_TO 35%, EXCAVATE 40%, MINE_ORE 20%, RETURN_TO 5%
        # Allow 5% tolerance
        total = float(n_samples)
        assert abs(counts[TaskType.MOVE_TO] / total - 0.35) < 0.05
        assert abs(counts[TaskType.EXCAVATE] / total - 0.40) < 0.05
        assert abs(counts[TaskType.MINE_ORE] / total - 0.20) < 0.05
        assert abs(counts[TaskType.RETURN_TO] / total - 0.05) < 0.05


# ── Test 6: Occupancy consistency (B2, B3) ───────────────────────────

class TestOccupancyConsistency:
    """Verify occupancy grid matches agent positions after random steps."""

    def _verify_occupancy(self, sw: SharedWorld) -> None:
        """Assert every registered agent's position matches occupancy."""
        for aid, turtle in sw._agents.items():
            pos = turtle.position
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            occ = sw._occupancy[x, y, z]
            assert occ == aid, (
                f"Agent {aid} at ({x},{y},{z}) but occupancy={occ}"
            )

    def test_register_deregister_cycle(self) -> None:
        sw = _make_shared_world(max_agents=4)
        pos = np.array([16, 16, 16], dtype=np.int32)
        sw[16, 16, 16] = BlockType.AIR
        t = Turtle(position=pos)
        assert sw.register_agent(0, t)
        self._verify_occupancy(sw)

        sw.deregister_agent(0)
        assert sw._occupancy[16, 16, 16] == -1
        assert 0 not in sw._agents

    def test_move_agent_success(self) -> None:
        sw = _make_shared_world(max_agents=4)
        old = np.array([16, 16, 16], dtype=np.int32)
        new = np.array([16, 17, 16], dtype=np.int32)
        sw[16, 16, 16] = BlockType.AIR
        sw[16, 17, 16] = BlockType.AIR
        t = Turtle(position=old)
        sw.register_agent(0, t)

        t.position = new.copy()
        ok = sw.move_agent(0, old, new)
        assert ok
        assert sw._occupancy[16, 16, 16] == -1
        assert sw._occupancy[16, 17, 16] == 0

    def test_move_agent_collision_fails(self) -> None:
        sw = _make_shared_world(max_agents=4)
        sw[16, 16, 16] = BlockType.AIR
        sw[16, 17, 16] = BlockType.AIR
        t0 = Turtle(position=np.array([16, 16, 16], dtype=np.int32))
        t1 = Turtle(position=np.array([16, 17, 16], dtype=np.int32))
        sw.register_agent(0, t0)
        sw.register_agent(1, t1)

        # Agent 0 tries to move into agent 1's position
        ok = sw.move_agent(0, t0.position, t1.position)
        assert not ok
        # Occupancy must be unchanged
        assert sw._occupancy[16, 16, 16] == 0
        assert sw._occupancy[16, 17, 16] == 1


# ── Test 7: Spawn positions skip ore blocks (W5) ────────────────────

class TestSpawnPositionsSkipOres:
    """Verify spawn_positions() never destroys ore blocks."""

    def test_ore_blocks_preserved(self) -> None:
        from prospect_rl.config import ORE_TYPES

        sw = _make_shared_world(size=(32, 32, 32), seed=42)
        ore_set = {int(bt) for bt in ORE_TYPES}

        # Snapshot ore counts before spawning
        world_arr = np.array(sw._world)

        # Spawn many agents
        rng = np.random.default_rng(123)
        sw.spawn_positions(n=20, min_distance=2, rng=rng)

        # Count ores after spawning
        world_arr_after = np.array(sw._world)
        for bt in ore_set:
            before = np.count_nonzero(world_arr == bt)
            after = np.count_nonzero(world_arr_after == bt)
            assert after == before, (
                f"Ore {bt}: {before} blocks before spawn, {after} after"
            )


# ── Test 8: EMA normalization stability (B7) ────────────────────────

class TestEMANormalization:
    """Verify EMA-smoothed normalization tracks max with smoothing."""

    def test_ema_never_below_one(self) -> None:
        """EMA max expected should never drop below 1.0."""
        size = (32, 32, 32)
        sw = _make_shared_world(size=size, seed=42, max_agents=4)

        from prospect_rl.multiagent.belief_map import BeliefMap
        from prospect_rl.multiagent.coordinator.coordinator import Coordinator
        from prospect_rl.multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=size[1],
            apply_solid_correction=False,
            apply_calibration=False,
        )
        bm = BeliefMap(
            world_size=size, biome_map=sw._world.biome_map,
            prior=prior, chunk_size_xz=16,
        )
        pref = np.ones(NUM_ORE_TYPES, dtype=np.float32) / NUM_ORE_TYPES

        # Register an agent so plan() has something to work with
        pos = np.array([16, 16, 16], dtype=np.int32)
        sw[16, 16, 16] = BlockType.AIR
        t = Turtle(position=pos)
        sw.register_agent(0, t)

        coord = Coordinator(
            belief_map=bm, shared_world=sw, preference=pref,
            chunk_size_xz=16,
        )

        # Run multiple plan() calls
        for _ in range(5):
            coord.plan()
            assert coord._ema_max_expected >= 1.0, (
                f"EMA dropped below 1.0: {coord._ema_max_expected}"
            )
