"""Coordinator: graph construction, GNN inference, Hungarian matching.

Orchestrates the full coordinator pipeline:
1. Build heterogeneous graph from agent states + belief map chunks
2. Run GNN forward pass to get per-(agent, region) scores
3. Apply bounding box constraints
4. Hungarian matching for optimal 1:1 assignment
5. Convert assignments to TaskAssignment objects
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from prospect_rl.config import NUM_ORE_TYPES, get_ore_y_ranges
from prospect_rl.multiagent.coordinator.assignment import (
    BoundingBox,
    TaskAssignment,
    TaskType,
)
from prospect_rl.multiagent.coordinator.gnn import (
    AGENT_FEATURE_DIM,
    REGION_FEATURE_DIM,
    CoordinatorGNNSimple,
    _HAS_PYG,
)

if _HAS_PYG:
    from prospect_rl.multiagent.coordinator.gnn import CoordinatorGNN

if TYPE_CHECKING:
    from prospect_rl.multiagent.belief_map import BeliefMap, ChunkState
    from prospect_rl.multiagent.shared_world import SharedWorld


class Coordinator:
    """Central coordinator for multi-agent task assignment.

    Parameters
    ----------
    belief_map:
        Shared belief map.
    shared_world:
        Shared world with occupancy.
    preference:
        Global 8-dim ore preference vector.
    gnn:
        GNN model for score prediction. If None, uses CoordinatorGNNSimple.
    bounding_boxes:
        Optional spatial constraints for mining regions.
    mining_radius:
        Distance threshold for switching from A* to RL mode.
    chunk_size_xz:
        XZ size of chunks (must match belief map).
    """

    def __init__(
        self,
        belief_map: BeliefMap,
        shared_world: SharedWorld,
        preference: np.ndarray,
        gnn: torch.nn.Module | None = None,
        bounding_boxes: list[BoundingBox] | None = None,
        mining_radius: int = 8,
        chunk_size_xz: int = 16,
        congestion_radius: int = 3,
        excavate_ore_threshold: float = 1.0,
        use_full_gnn: bool = False,
        preference_blend_alpha: float = 0.5,
    ) -> None:
        self._belief_map = belief_map
        self._shared_world = shared_world
        self._preference = preference
        self._preference_blend_alpha = preference_blend_alpha
        self._bounding_boxes = bounding_boxes or []
        self._mining_radius = mining_radius
        self._chunk_size_xz = chunk_size_xz
        self._congestion_radius = congestion_radius
        self._excavate_ore_threshold = excavate_ore_threshold

        # GNN (use full GATv2Conv if PyG available, else simple MLP fallback)
        if gnn is not None:
            self._gnn = gnn
        elif use_full_gnn and _HAS_PYG:
            self._gnn = CoordinatorGNN()
        else:
            self._gnn = CoordinatorGNNSimple()

        self._device = next(self._gnn.parameters()).device

        # Alpha for blended assignment (1.0 = pure GNN, 0.0 = pure heuristic)
        self._alpha: float = 1.0

        # Current assignments
        self._assignments: dict[int, TaskAssignment] = {}
        self._step_since_replan: int = 0

        # Last plan intermediates (for REINFORCE training in Phase 3)
        self._last_x_dict: dict[str, torch.Tensor] | None = None
        self._last_edge_index_dict: dict | None = None
        self._last_row_idx: np.ndarray | None = None
        self._last_col_idx: np.ndarray | None = None

        # EMA-smoothed max for stable region feature normalization
        self._ema_max_expected: float = 1.0
        self._ema_norm_alpha: float = 0.1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, step: int = 0) -> dict[int, TaskAssignment]:
        """Run the full coordinator pipeline and return assignments.

        Returns a dict mapping agent_id to TaskAssignment.
        """
        agent_positions = self._shared_world.get_agent_positions()
        if not agent_positions:
            return {}

        agent_ids = sorted(agent_positions.keys())
        chunks = self._belief_map.get_all_chunks()

        if not chunks:
            return {}

        # Build features
        agent_features = self._build_agent_features(agent_ids, agent_positions)
        chunk_keys = sorted(chunks.keys())
        region_features = self._build_region_features(chunk_keys, chunks)

        # Build edge indices
        edge_index_dict = self._build_edge_indices(
            agent_ids, agent_positions, chunk_keys, chunks,
        )

        # GNN forward pass
        x_dict = {
            "agent": torch.tensor(agent_features, dtype=torch.float32, device=self._device),
            "region": torch.tensor(region_features, dtype=torch.float32, device=self._device),
        }

        with torch.no_grad():
            scores = self._gnn(x_dict, edge_index_dict)  # (N_a, N_r)

        # Blend GNN and heuristic scores (Phase 4)
        if self._alpha < 1.0:
            heuristic_scores = self._compute_heuristic_scores(
                agent_ids, agent_positions, chunk_keys, chunks,
            )
            scores = self._alpha * scores + (1.0 - self._alpha) * heuristic_scores

        # Apply bounding box hard mask
        if self._bounding_boxes:
            scores = self._apply_bbox_mask(scores, chunk_keys)

        # Hungarian matching
        assignments = self._hungarian_match(
            scores, agent_ids, chunk_keys, chunks, agent_positions,
        )

        # Store intermediates for REINFORCE training
        self._last_x_dict = x_dict
        self._last_edge_index_dict = edge_index_dict

        self._assignments = assignments
        self._step_since_replan = 0
        return assignments

    def get_assignment(self, agent_id: int) -> TaskAssignment | None:
        """Get the current assignment for an agent."""
        return self._assignments.get(agent_id)

    def increment_step(self) -> None:
        """Increment the step counter since last replan."""
        self._step_since_replan += 1

    @property
    def step_since_replan(self) -> int:
        return self._step_since_replan

    @property
    def alpha(self) -> float:
        """Blending factor: 1.0 = pure GNN, 0.0 = pure heuristic."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = max(0.0, min(1.0, value))

    # ------------------------------------------------------------------
    # Heuristic scoring (for Phase 4 blending)
    # ------------------------------------------------------------------

    def _compute_heuristic_scores(
        self,
        agent_ids: list[int],
        agent_positions: dict[int, np.ndarray],
        chunk_keys: list[tuple[int, int]],
        chunks: dict[tuple[int, int], ChunkState],
    ) -> torch.Tensor:
        """Compute heuristic scores with distribution deficit awareness.

        Base score = dot(expected_remaining, preference) / distance.
        Deficit boost: under-mined ore types get higher weight,
        steering agents toward under-represented targets.

        Returns (N_agents, N_regions) tensor on self._device.
        """
        n_agents = len(agent_ids)
        n_regions = len(chunk_keys)
        sx, sy, sz = self._shared_world.shape

        # Compute team mining deficit
        team_mined = self._compute_team_mining_progress(chunks)
        pref = self._preference
        pref_sum = float(np.sum(pref))
        deficit_weight = pref.copy()

        if pref_sum > 0 and float(np.sum(team_mined)) > 0:
            target = pref / pref_sum
            weighted_mined = team_mined.copy()
            mined_sum = float(np.sum(weighted_mined))
            if mined_sum > 0:
                actual = weighted_mined / mined_sum
                # Deficit: how much each ore is under-mined
                deficit = np.maximum(target - actual, 0.0)
                d_sum = float(np.sum(deficit))
                if d_sum > 0:
                    deficit /= d_sum
                    # Blend: 70% preference + 30% deficit
                    deficit_weight = (
                        0.7 * target + 0.3 * deficit
                    ).astype(np.float32)

        scores = np.zeros(
            (n_agents, n_regions), dtype=np.float32,
        )
        for i, aid in enumerate(agent_ids):
            apos = agent_positions[aid]
            for j, key in enumerate(chunk_keys):
                chunk = chunks[key]
                ore_value = float(
                    np.dot(
                        chunk.expected_remaining,
                        deficit_weight,
                    )
                )
                cx = int(
                    (key[0] + 0.5) * self._chunk_size_xz,
                )
                cz = int(
                    (key[1] + 0.5) * self._chunk_size_xz,
                )
                cy = sy // 2
                dist = (
                    abs(int(apos[0]) - cx)
                    + abs(int(apos[1]) - cy)
                    + abs(int(apos[2]) - cz)
                )
                scores[i, j] = ore_value / (dist + 1.0)

        return torch.tensor(
            scores, dtype=torch.float32, device=self._device,
        )

    # ------------------------------------------------------------------
    # Per-agent preference & distribution awareness
    # ------------------------------------------------------------------

    def _compute_agent_preference(
        self,
        chunk: ChunkState,
        team_preference: np.ndarray,
    ) -> np.ndarray:
        """Blend team preference with regional ore availability.

        alpha=1.0 → pure team preference (global).
        alpha=0.0 → pure regional availability (local).
        Filtered to team-desired ores only, renormalized.
        """
        alpha = self._preference_blend_alpha

        # Regional availability from chunk expected remaining
        regional = chunk.expected_remaining.copy()
        reg_sum = float(np.sum(regional))
        if reg_sum > 0:
            regional /= reg_sum
        else:
            regional = np.ones(
                NUM_ORE_TYPES, dtype=np.float32,
            ) / NUM_ORE_TYPES

        # Blend
        blended = (
            alpha * team_preference
            + (1.0 - alpha) * regional
        )

        # Filter to team-desired ores only
        mask = team_preference > 0
        blended[~mask] = 0.0
        blend_sum = float(np.sum(blended))
        if blend_sum > 0:
            blended /= blend_sum
        else:
            blended = team_preference.copy()

        return blended.astype(np.float32)

    def _compute_team_mining_progress(
        self,
        chunks: dict[tuple[int, int], ChunkState],
    ) -> np.ndarray:
        """Aggregate mined ore counts across all chunks.

        Returns shape (NUM_ORE_TYPES,) total mined.
        """
        total = np.zeros(NUM_ORE_TYPES, dtype=np.float64)
        for chunk in chunks.values():
            total += chunk.mined_ore_counts
        return total

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_agent_features(
        self,
        agent_ids: list[int],
        agent_positions: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Build agent node features matrix.

        Shape: (N_agents, AGENT_FEATURE_DIM=24)
        """
        sx, sy, sz = self._shared_world.shape
        world_diag = math.sqrt(sx * sx + sy * sy + sz * sz)

        features = np.zeros((len(agent_ids), AGENT_FEATURE_DIM), dtype=np.float32)

        for i, aid in enumerate(agent_ids):
            pos = agent_positions[aid]
            turtle = self._shared_world.get_agent(aid)

            # Position (normalized)
            features[i, 0:3] = pos / np.array([sx, sy, sz], dtype=np.float32)

            # Fuel fraction
            if turtle is not None:
                features[i, 3] = turtle.fuel / max(turtle.max_fuel, 1)

                # Inventory (8 ore counts, normalized)
                from prospect_rl.config import ORE_TYPES
                for j, ore_bt in enumerate(ORE_TYPES):
                    features[i, 4 + j] = min(
                        1.0,
                        turtle.inventory.get(int(ore_bt), 0) / 100.0,
                    )

            # Current assignment (normalized target position)
            existing = self._assignments.get(aid)
            if existing is not None:
                features[i, 12:15] = (
                    existing.target_position / np.array([sx, sy, sz], dtype=np.float32)
                )

            # Steps since replan (normalized)
            features[i, 15] = min(1.0, self._step_since_replan / 100.0)

            # Preference (8 dims) — per-agent if assigned
            existing_assign = self._assignments.get(aid)
            if existing_assign is not None:
                features[i, 16:24] = existing_assign.ore_preference
            else:
                features[i, 16:24] = self._preference

        return features

    def _build_region_features(
        self,
        chunk_keys: list[tuple[int, int]],
        chunks: dict[tuple[int, int], ChunkState],
    ) -> np.ndarray:
        """Build region node features matrix.

        Shape: (N_regions, REGION_FEATURE_DIM=20)
        """
        sx, sy, sz = self._shared_world.shape

        features = np.zeros(
            (len(chunk_keys), REGION_FEATURE_DIM), dtype=np.float32,
        )

        # EMA-smoothed max for stable cross-region normalization.
        # Using a raw per-plan max causes non-stationary features that
        # destabilize REINFORCE training as exploration progresses.
        current_max = 1.0
        for key in chunk_keys:
            local_max = float(np.max(chunks[key].expected_remaining))
            if local_max > current_max:
                current_max = local_max
        self._ema_max_expected = (
            self._ema_norm_alpha * current_max
            + (1.0 - self._ema_norm_alpha) * self._ema_max_expected
        )
        global_max_expected = max(self._ema_max_expected, 1.0)

        for i, key in enumerate(chunk_keys):
            chunk = chunks[key]

            # Center position (normalized)
            cx = (key[0] + 0.5) * self._chunk_size_xz / sx
            cy = 0.5  # Middle of world height
            cz = (key[1] + 0.5) * self._chunk_size_xz / sz
            features[i, 0:3] = [cx, cy, cz]

            # Expected remaining ore (8 dims, log-normalized)
            features[i, 3:11] = np.log1p(chunk.expected_remaining) / np.log1p(global_max_expected)

            # Information gain (normalized)
            # Use a simplified metric: unexplored fraction
            features[i, 11] = 1.0 - chunk.explored_frac

            # Explored fraction
            features[i, 12] = chunk.explored_frac

            # Biome one-hot (5 dims)
            from prospect_rl.config import NUM_BIOME_TYPES
            biome_id = min(chunk.biome_id, NUM_BIOME_TYPES - 1)
            features[i, 13 + biome_id] = 1.0

            # Assigned agent count
            features[i, 18] = min(1.0, chunk.assigned_agent_count / 5.0)

            # Bounding box mask (1 if in any box, 0 if not)
            if not self._bounding_boxes:
                features[i, 19] = 1.0  # No constraints = all valid
            else:
                center = (
                    int((key[0] + 0.5) * self._chunk_size_xz),
                    sy // 2,
                    int((key[1] + 0.5) * self._chunk_size_xz),
                )
                for bbox in self._bounding_boxes:
                    if bbox.contains(*center):
                        features[i, 19] = 1.0
                        break

        return features

    def _build_edge_indices(
        self,
        agent_ids: list[int],
        agent_positions: dict[int, np.ndarray],
        chunk_keys: list[tuple[int, int]],
        chunks: dict[tuple[int, int], ChunkState],
    ) -> dict:
        """Build edge index dict for the heterogeneous graph."""
        n_agents = len(agent_ids)
        n_regions = len(chunk_keys)

        # Agent → Region: fully connected bipartite
        src_ar = []
        dst_ar = []
        for i in range(n_agents):
            for j in range(n_regions):
                src_ar.append(i)
                dst_ar.append(j)

        # Region → Agent: reverse edges
        src_ra = dst_ar.copy()
        dst_ra = src_ar.copy()

        # Agent → Agent: nearby agents
        src_aa = []
        dst_aa = []
        agent_pos_list = [agent_positions[aid] for aid in agent_ids]
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = int(np.sum(np.abs(agent_pos_list[i] - agent_pos_list[j])))
                if dist <= 2 * self._congestion_radius:
                    src_aa.extend([i, j])
                    dst_aa.extend([j, i])

        # Region → Region: adjacent chunks
        src_rr = []
        dst_rr = []
        chunk_key_to_idx = {k: i for i, k in enumerate(chunk_keys)}
        for i, key in enumerate(chunk_keys):
            cx, cz = key
            for dcx, dcz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (cx + dcx, cz + dcz)
                if neighbor in chunk_key_to_idx:
                    j = chunk_key_to_idx[neighbor]
                    src_rr.append(i)
                    dst_rr.append(j)

        device = self._device

        def _to_edge_tensor(src: list, dst: list) -> torch.Tensor:
            if not src:
                return torch.zeros((2, 0), dtype=torch.long, device=device)
            return torch.tensor([src, dst], dtype=torch.long, device=device)

        return {
            ("agent", "considers", "region"): _to_edge_tensor(src_ar, dst_ar),
            ("region", "considered_by", "agent"): _to_edge_tensor(src_ra, dst_ra),
            ("agent", "near", "agent"): _to_edge_tensor(src_aa, dst_aa),
            ("region", "adjacent", "region"): _to_edge_tensor(src_rr, dst_rr),
        }

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _apply_bbox_mask(
        self, scores: torch.Tensor, chunk_keys: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Set scores to -inf for regions outside all bounding boxes."""
        sx, sy, sz = self._shared_world.shape
        for j, key in enumerate(chunk_keys):
            center_x = int((key[0] + 0.5) * self._chunk_size_xz)
            center_z = int((key[1] + 0.5) * self._chunk_size_xz)
            in_any = False
            for bbox in self._bounding_boxes:
                if bbox.contains(center_x, sy // 2, center_z):
                    in_any = True
                    break
            if not in_any:
                scores[:, j] = float("-inf")
        return scores

    def _hungarian_match(
        self,
        scores: torch.Tensor,
        agent_ids: list[int],
        chunk_keys: list[tuple[int, int]],
        chunks: dict[tuple[int, int], ChunkState],
        agent_positions: dict[int, np.ndarray],
    ) -> dict[int, TaskAssignment]:
        """Run Hungarian matching and build TaskAssignment objects."""
        from scipy.optimize import linear_sum_assignment

        cost_matrix = -scores.detach().cpu().numpy()

        # Handle case where there are more agents than regions
        n_agents = len(agent_ids)
        n_regions = len(chunk_keys)

        if n_agents == 0 or n_regions == 0:
            return {}

        # Pad cost matrix if needed (more agents than regions)
        if n_agents > n_regions:
            pad = np.full(
                (n_agents, n_agents - n_regions),
                1e6,  # High cost for "no assignment"
                dtype=cost_matrix.dtype,
            )
            cost_matrix = np.concatenate([cost_matrix, pad], axis=1)

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        # Store only valid (non-dummy) assignments for REINFORCE training.
        # When n_agents > n_regions, dummy columns (>= n_regions) are padded;
        # these must be excluded to prevent IndexError in the trainer.
        valid = col_idx < n_regions
        self._last_row_idx = row_idx[valid]
        self._last_col_idx = col_idx[valid]

        assignments = {}
        sx, sy, sz = self._shared_world.shape

        # Reset assigned agent counts
        for chunk in chunks.values():
            chunk.assigned_agent_count = 0

        for r, c in zip(row_idx, col_idx):
            if c >= n_regions:
                continue  # Dummy assignment, skip

            aid = agent_ids[r]
            chunk_key = chunk_keys[c]
            chunk = chunks[chunk_key]

            # Determine task type
            task_type = self._determine_task_type(
                chunk, agent_positions.get(aid),
            )

            # Compute target position (chunk center)
            target_x = int((chunk_key[0] + 0.5) * self._chunk_size_xz)
            target_z = int((chunk_key[1] + 0.5) * self._chunk_size_xz)

            # Y target: use ore preference to pick best Y level
            target_y = self._compute_target_y(chunk)

            target_pos = np.array(
                [min(target_x, sx - 1), min(target_y, sy - 1), min(target_z, sz - 1)],
                dtype=np.int32,
            )

            # Build bounding box from chunk with ore-based Y-range
            y_ranges = get_ore_y_ranges(sy)
            y_min_ore, y_max_ore = sy, 0
            for oi in range(NUM_ORE_TYPES):
                if self._preference[oi] > 0:
                    y_min_ore = min(y_min_ore, int(y_ranges[oi][0]))
                    y_max_ore = max(y_max_ore, int(y_ranges[oi][1]))
            if y_min_ore >= y_max_ore:
                y_min_ore, y_max_ore = 0, sy - 1
            y_min_clamped = max(0, y_min_ore - 5)
            y_max_clamped = min(sy - 1, y_max_ore + 5)

            bbox = BoundingBox(
                x_min=chunk_key[0] * self._chunk_size_xz,
                x_max=min((chunk_key[0] + 1) * self._chunk_size_xz - 1, sx - 1),
                z_min=chunk_key[1] * self._chunk_size_xz,
                z_max=min((chunk_key[1] + 1) * self._chunk_size_xz - 1, sz - 1),
                y_min=y_min_clamped,
                y_max=y_max_clamped,
            )

            agent_pref = self._compute_agent_preference(
                chunk, self._preference,
            )

            assignments[aid] = TaskAssignment(
                agent_id=aid,
                task_type=task_type,
                target_position=target_pos,
                bounding_box=bbox,
                ore_preference=agent_pref,
                region_index=c,
            )

            chunk.assigned_agent_count += 1

        return assignments

    def _determine_task_type(
        self,
        chunk: ChunkState,
        agent_pos: np.ndarray | None,
    ) -> TaskType:
        """Determine the appropriate task type for a chunk."""
        if chunk.explored_frac < 0.3:
            return TaskType.EXCAVATE

        # Check if there's known ore in the chunk
        if np.any(chunk.expected_remaining > self._excavate_ore_threshold):
            return TaskType.MINE_ORE

        return TaskType.EXCAVATE

    def _compute_target_y(self, chunk: ChunkState) -> int:
        """Compute the target Y level based on preference and ore ranges."""
        sy = self._shared_world.shape[1]
        y_ranges = get_ore_y_ranges(sy)

        # Weighted average of preferred ore Y-ranges
        total_weight = 0.0
        weighted_y = 0.0
        for i in range(NUM_ORE_TYPES):
            w = float(self._preference[i])
            if w > 0:
                y_min, y_max = y_ranges[i]
                y_mid = (y_min + y_max) / 2.0
                weighted_y += w * y_mid
                total_weight += w

        if total_weight > 0:
            return int(weighted_y / total_weight)
        return sy // 2
