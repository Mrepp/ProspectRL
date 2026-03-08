"""Bernoulli voxel belief map for multi-agent ore estimation.

Each voxel ``v`` has per-ore-type probability ``p_v[ore_type]``.
Sparse storage: only observed/modified voxels are stored; unobserved
voxels implicitly have p_v = prior(y, biome).

Supports Bayesian updates from telemetry events, cluster propagation
on ore discovery, and chunk-level expected remaining ore estimates.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from prospect_rl.config import NUM_ORE_TYPES, ORE_INDEX as _ORE_INDEX

if TYPE_CHECKING:
    from prospect_rl.multiagent.geological_prior import AnalyticalPrior
    from prospect_rl.multiagent.telemetry import TelemetryEvent


class ChunkStatus(IntEnum):
    """State of a chunk's exploration."""

    UNKNOWN = 0              # no voxels observed
    PARTIALLY_EXPLORED = 1   # some voxels observed, ore may remain
    EXPLORED = 2             # majority of voxels observed
    EXHAUSTED = 3            # E_remaining < epsilon for all ore types


class ChunkState:
    """Per-chunk tracking for belief map.

    Parameters
    ----------
    chunk_coords:
        (cx, cz) chunk coordinates.
    total_voxels:
        Total number of voxels in this chunk column.
    """

    __slots__ = (
        "chunk_coords", "status", "biome_id", "total_voxels",
        "observed_voxels", "mined_ore_counts", "expected_remaining",
        "last_visit_step", "assigned_agent_count",
    )

    def __init__(
        self,
        chunk_coords: tuple[int, int],
        total_voxels: int,
        biome_id: int = 0,
    ) -> None:
        self.chunk_coords = chunk_coords
        self.status = ChunkStatus.UNKNOWN
        self.biome_id = biome_id
        self.total_voxels = total_voxels
        self.observed_voxels = 0
        self.mined_ore_counts = np.zeros(NUM_ORE_TYPES, dtype=np.int32)
        self.expected_remaining = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        self.last_visit_step = -1
        self.assigned_agent_count = 0

    @property
    def explored_frac(self) -> float:
        if self.total_voxels == 0:
            return 0.0
        return self.observed_voxels / self.total_voxels

    def update_status(self, depletion_epsilon: float = 0.1) -> None:
        """Update chunk status based on exploration and depletion."""
        if self.observed_voxels == 0:
            self.status = ChunkStatus.UNKNOWN
        elif np.all(self.expected_remaining < depletion_epsilon):
            self.status = ChunkStatus.EXHAUSTED
        elif self.explored_frac > 0.8:
            self.status = ChunkStatus.EXPLORED
        else:
            self.status = ChunkStatus.PARTIALLY_EXPLORED


class BeliefMap:
    """Bernoulli voxel belief map with sparse observation storage.

    Parameters
    ----------
    world_size:
        (sx, sy, sz) world dimensions.
    biome_map:
        2D biome map of shape (sx, sz).
    prior:
        Analytical geological prior for unobserved voxels.
    chunk_size_xz:
        XZ size of chunks for aggregation.
    cluster_strength:
        Lambda — boost to neighbors on ore find.
    depletion_epsilon:
        Threshold for marking chunks as exhausted.
    invalidation_radius:
        Blocks around changed voxels to invalidate.
    """

    def __init__(
        self,
        world_size: tuple[int, int, int],
        biome_map: np.ndarray,
        prior: AnalyticalPrior,
        chunk_size_xz: int = 16,
        cluster_strength: float = 0.3,
        depletion_epsilon: float = 0.1,
        invalidation_radius: int = 1,
    ) -> None:
        self._world_size = world_size
        self._biome_map = biome_map
        self._prior = prior
        self._chunk_size_xz = chunk_size_xz
        self._cluster_strength = cluster_strength
        self._depletion_epsilon = depletion_epsilon
        self._invalidation_radius = invalidation_radius

        # Sparse voxel storage: only store p_v for observed/modified voxels
        # Key: (x, y, z), Value: ndarray(8,) of per-ore probabilities
        self._observed: dict[tuple[int, int, int], np.ndarray] = {}
        # Track which voxels have been confirmed observed (vs cluster-boosted)
        self._confirmed: set[tuple[int, int, int]] = set()

        # Initialize chunk states
        sx, sy, sz = world_size
        self._chunks: dict[tuple[int, int], ChunkState] = {}
        num_cx = max(1, (sx + chunk_size_xz - 1) // chunk_size_xz)
        num_cz = max(1, (sz + chunk_size_xz - 1) // chunk_size_xz)

        for cx in range(num_cx):
            for cz in range(num_cz):
                # Determine dominant biome for this chunk
                x0 = cx * chunk_size_xz
                z0 = cz * chunk_size_xz
                x1 = min(x0 + chunk_size_xz, sx)
                z1 = min(z0 + chunk_size_xz, sz)
                biome_slice = biome_map[x0:x1, z0:z1]
                if biome_slice.size > 0:
                    biome_id = int(np.bincount(
                        biome_slice.ravel().astype(np.int32),
                        minlength=1,
                    ).argmax())
                else:
                    biome_id = 0

                chunk_volume = (x1 - x0) * sy * (z1 - z0)
                chunk = ChunkState(
                    chunk_coords=(cx, cz),
                    total_voxels=chunk_volume,
                    biome_id=biome_id,
                )
                # Initialize expected remaining from prior
                chunk.expected_remaining = prior.query_chunk(
                    cx, cz, biome_map, chunk_size_xz,
                )
                self._chunks[(cx, cz)] = chunk

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_belief(self, x: int, y: int, z: int) -> np.ndarray:
        """Return P(ore_type) at position (x, y, z).

        Returns shape ``(8,)`` — per-ore probability.
        """
        pos = (x, y, z)
        if pos in self._observed:
            return self._observed[pos].copy()
        # Unobserved: use prior
        sx, sz = self._biome_map.shape
        bx = min(max(0, x), sx - 1)
        bz = min(max(0, z), sz - 1)
        biome_id = int(self._biome_map[bx, bz])
        return self._prior.query(y, biome_id)

    def get_chunk_state(self, cx: int, cz: int) -> ChunkState | None:
        """Return the chunk state at chunk coordinates (cx, cz)."""
        return self._chunks.get((cx, cz))

    def get_all_chunks(self) -> dict[tuple[int, int], ChunkState]:
        """Return all chunk states."""
        return self._chunks

    def is_observed(self, x: int, y: int, z: int) -> bool:
        """Check if a voxel has been directly observed."""
        return (x, y, z) in self._confirmed

    # ------------------------------------------------------------------
    # Information gain
    # ------------------------------------------------------------------

    def chunk_information_gain(self, cx: int, cz: int) -> float:
        """Compute the Shannon entropy (information to gain) for a chunk.

        Higher values mean more uncertainty = more value in exploring.
        """
        chunk = self._chunks.get((cx, cz))
        if chunk is None:
            return 0.0

        sx, sy, sz = self._world_size
        x0 = cx * self._chunk_size_xz
        z0 = cz * self._chunk_size_xz
        x1 = min(x0 + self._chunk_size_xz, sx)
        z1 = min(z0 + self._chunk_size_xz, sz)

        total_entropy = 0.0
        count = 0
        for x in range(x0, x1):
            for z in range(z0, z1):
                for y in range(sy):
                    pos = (x, y, z)
                    if pos in self._confirmed:
                        continue  # Already observed, no information to gain
                    p = self.get_belief(x, y, z)
                    for p_ore in p:
                        if 0 < p_ore < 1:
                            total_entropy += (
                                -p_ore * math.log(p_ore + 1e-10)
                                - (1 - p_ore) * math.log(1 - p_ore + 1e-10)
                            )
                    count += 1

        return total_entropy

    # ------------------------------------------------------------------
    # Updates from telemetry
    # ------------------------------------------------------------------

    def process_events(
        self, events: list[TelemetryEvent], step: int = 0,
    ) -> None:
        """Process a batch of telemetry events and update beliefs."""
        from prospect_rl.multiagent.telemetry import TelemetryEventType

        for event in events:
            pos = event.position
            x, y, z = pos

            if event.event_type == TelemetryEventType.BLOCK_OBSERVED:
                self._handle_observation(x, y, z, event.block_type, step)

            elif event.event_type == TelemetryEventType.BLOCK_REMOVED:
                self._handle_removal(x, y, z, event.block_type, event.ore_type, step)

            elif event.event_type == TelemetryEventType.BLOCK_ADDED:
                self._handle_addition(x, y, z, event.block_type, step)

            elif event.event_type in (
                TelemetryEventType.BLOCK_CHANGED,
                TelemetryEventType.PATH_BLOCKED,
            ):
                self._handle_invalidation(x, y, z, step)

    def _handle_observation(
        self, x: int, y: int, z: int, block_type: int, step: int,
    ) -> None:
        """Handle a block observation (3-block inspect or dig-through)."""
        pos = (x, y, z)

        # Capture old state for incremental update
        was_confirmed = pos in self._confirmed
        old_belief = self._observed.get(pos)

        if block_type in _ORE_INDEX:
            # Ore found
            ore_idx = _ORE_INDEX[block_type]
            p = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
            p[ore_idx] = 1.0
            self._observed[pos] = p
            self._confirmed.add(pos)
            self._propagate_cluster(x, y, z, ore_idx)
        else:
            # Non-ore: clear all ore probabilities
            p = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
            self._observed[pos] = p
            self._confirmed.add(pos)

        self._update_chunk_incremental(
            x, y, z, step, old_belief, p, was_confirmed, True,
        )

    def _handle_removal(
        self,
        x: int, y: int, z: int,
        block_type: int,
        ore_type: int | None,
        step: int,
    ) -> None:
        """Handle a block being mined."""
        pos = (x, y, z)

        # Capture old state for incremental update
        was_confirmed = pos in self._confirmed
        old_belief = self._observed.get(pos)

        p = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        self._observed[pos] = p
        self._confirmed.add(pos)

        # Update chunk mined counts
        if ore_type is not None:
            chunk_key = self._pos_to_chunk(x, z)
            chunk = self._chunks.get(chunk_key)
            if chunk is not None:
                chunk.mined_ore_counts[ore_type] += 1

        self._update_chunk_incremental(
            x, y, z, step, old_belief, p, was_confirmed, True,
        )

    def _handle_addition(
        self, x: int, y: int, z: int, block_type: int, step: int,
    ) -> None:
        """Handle a block appearing (gravel fall, player placement)."""
        pos = (x, y, z)

        # Capture old state for incremental update
        was_confirmed = pos in self._confirmed
        old_belief = self._observed.get(pos)

        if block_type in _ORE_INDEX:
            ore_idx = _ORE_INDEX[block_type]
            p = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
            p[ore_idx] = 1.0
            self._observed[pos] = p
        else:
            p = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
            self._observed[pos] = p
        self._confirmed.add(pos)

        self._update_chunk_incremental(
            x, y, z, step, old_belief, p, was_confirmed, True,
        )

    def _handle_invalidation(
        self, x: int, y: int, z: int, step: int,
    ) -> None:
        """Invalidate belief for a voxel and its neighbors."""
        r = self._invalidation_radius
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    pos = (nx, ny, nz)
                    if pos in self._observed:
                        # Revert to prior
                        del self._observed[pos]
                        self._confirmed.discard(pos)

        self._recount_chunk(x, y, z, step)

    # ------------------------------------------------------------------
    # Cluster propagation
    # ------------------------------------------------------------------

    def _propagate_cluster(
        self, x: int, y: int, z: int, ore_idx: int,
    ) -> None:
        """Boost ore probability in unobserved neighbors when ore is found."""
        radius = self._prior.get_cluster_radius(ore_idx)
        lam = self._cluster_strength
        r_int = int(math.ceil(radius))
        sx, sy, sz = self._world_size

        for dx in range(-r_int, r_int + 1):
            for dy in range(-r_int, r_int + 1):
                for dz in range(-r_int, r_int + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if nx < 0 or ny < 0 or nz < 0:
                        continue
                    if nx >= sx or ny >= sy or nz >= sz:
                        continue

                    pos = (nx, ny, nz)
                    if pos in self._confirmed:
                        continue  # Don't modify confirmed observations

                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist > radius:
                        continue

                    boost = lam * math.exp(-dist / radius)
                    p = self.get_belief(nx, ny, nz)
                    p[ore_idx] = min(1.0, p[ore_idx] + boost)
                    self._observed[pos] = p

    # ------------------------------------------------------------------
    # Chunk helpers
    # ------------------------------------------------------------------

    def _pos_to_chunk(self, x: int, z: int) -> tuple[int, int]:
        """Convert world XZ coordinates to chunk coordinates."""
        return (x // self._chunk_size_xz, z // self._chunk_size_xz)

    def _update_chunk_incremental(
        self,
        x: int, y: int, z: int,
        step: int,
        old_belief: np.ndarray | None,
        new_belief: np.ndarray,
        was_confirmed: bool,
        is_confirmed: bool,
    ) -> None:
        """Incrementally update chunk metadata for a single voxel change.

        Parameters
        ----------
        old_belief:
            Previous belief vector for this voxel (None if was unobserved
            and using prior, in which case we compute it).
        new_belief:
            New belief vector for this voxel.
        was_confirmed:
            Whether the voxel was in ``_confirmed`` before the update.
        is_confirmed:
            Whether the voxel is in ``_confirmed`` after the update.
        """
        chunk_key = self._pos_to_chunk(x, z)
        chunk = self._chunks.get(chunk_key)
        if chunk is None:
            return

        # Delta for observed_voxels
        if is_confirmed and not was_confirmed:
            chunk.observed_voxels += 1
        elif was_confirmed and not is_confirmed:
            chunk.observed_voxels = max(0, chunk.observed_voxels - 1)

        # Delta for expected_remaining: subtract old, add new
        if old_belief is None:
            _sx, _sz = self._biome_map.shape
            _bx = min(max(0, x), _sx - 1)
            _bz = min(max(0, z), _sz - 1)
            old_belief = self._prior.query(y, int(self._biome_map[_bx, _bz]))
        chunk.expected_remaining -= old_belief
        chunk.expected_remaining += new_belief
        # Clamp to non-negative
        np.maximum(chunk.expected_remaining, 0.0, out=chunk.expected_remaining)

        chunk.last_visit_step = step
        chunk.update_status(self._depletion_epsilon)

    def _recount_chunk(
        self, x: int, y: int, z: int, step: int,
    ) -> None:
        """Full recount of chunk metadata (used by invalidation and debug).

        O(chunk_size_xz^2 * world_height) per call.  Only triggered by
        rare BLOCK_CHANGED / PATH_BLOCKED events.  The common-case
        updates use ``_update_chunk_incremental`` which is O(1).
        """
        chunk_key = self._pos_to_chunk(x, z)
        chunk = self._chunks.get(chunk_key)
        if chunk is None:
            return

        sx, sy, sz = self._world_size
        x0 = chunk_key[0] * self._chunk_size_xz
        z0 = chunk_key[1] * self._chunk_size_xz
        x1 = min(x0 + self._chunk_size_xz, sx)
        z1 = min(z0 + self._chunk_size_xz, sz)

        obs_count = 0
        expected = np.zeros(NUM_ORE_TYPES, dtype=np.float32)

        for cx in range(x0, x1):
            for cz in range(z0, z1):
                for cy in range(sy):
                    pos = (cx, cy, cz)
                    if pos in self._confirmed:
                        obs_count += 1
                        if pos in self._observed:
                            expected += self._observed[pos]
                    else:
                        expected += self.get_belief(cx, cy, cz)

        chunk.observed_voxels = obs_count
        chunk.expected_remaining = expected
        chunk.last_visit_step = step
        chunk.update_status(self._depletion_epsilon)
