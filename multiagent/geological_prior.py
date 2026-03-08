"""Analytical geological prior for ore distribution.

Derives P(ore_type | y, biome) from parsed vanilla 1.21.11 worldgen JSON.
Each placed feature contributes::

    E[attempts] * P_eff(y) * expected_blocks(size) / 256 * repl_frac(y)
                * (1 - discard * P_adj_air(y))

corrected by three empirically-measured terrain profiles:

1. **Height mass loss**: Distributions extending beyond world bounds (e.g.
   diamond below Y=-64, emerald above Y=320) waste attempts; the effective
   PMF sums to <1.0 for these ores.
2. **Replaceable fraction**: P(block in ``stone_ore_replaceables`` | y, biome)
   — the fraction of blocks that MC can replace with ore.  Correctly handles
   bedrock (solid but not replaceable) and dirt/gravel/clay.
3. **Expected blocks per vein**: MC's ``size`` parameter controls blob
   geometry, not block count.  Small sizes produce fewer blocks than the
   parameter value due to ellipsoid discretization.  Lookup table derived
   from empirical MC chunk analysis.
4. **Air-exposure discard**: ``discard_chance_on_air_exposure`` discards
   blocks adjacent to air, modeled using measured P(adjacent_to_air | solid).

An optional empirical calibration table (per-ore per-Y-band multiplicative
factors) can further refine the analytical predictions.

Falls back to ``ORE_SPAWN_CONFIGS``-based computation if the JSON files are
unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from prospect_rl.config import (
    MC_Y_MIN,
    MC_Y_RANGE,
    NUM_BIOME_TYPES,
    NUM_ORE_TYPES,
    ORE_SPAWN_CONFIGS,
    ORE_TYPE_CONFIGS,
    ORE_TYPES,
    OreSpawnConfig,
    get_ore_y_ranges,
)

# Maps MC ore block names to our 0-7 ore index.  Both stone and deepslate
# variants map to the same index.
_MC_BLOCK_TO_ORE_INDEX: dict[str, int] = {}
for _i, _bt in enumerate(ORE_TYPES):
    _name = _bt.name.lower().replace("_ore", "")
    _MC_BLOCK_TO_ORE_INDEX[f"{_name}_ore"] = _i
    _MC_BLOCK_TO_ORE_INDEX[f"deepslate_{_name}_ore"] = _i

# MC world height used for the terrain profiles (384 = Overworld range).
_MC_WORLD_HEIGHT = 384

# ---------------------------------------------------------------------------
# Expected blocks per MC vein attempt
# ---------------------------------------------------------------------------

# MC's ``size`` parameter controls blob geometry (a capsule of overlapping
# ellipsoids along a random line), NOT the block count.  For small sizes the
# ellipsoids are too small to capture integer block centres, yielding fewer
# blocks than ``size``.  For large sizes, sphere overlap produces more blocks.
#
# Values derived from 242 real MC 1.21 chunks by back-solving:
#   actual_ores_per_col = sum_y[ attempts * P_eff(y) * geo(size) / 256
#                                * repl_frac(y) * (1 - discard * P_adj_air) ]
# Single-size ores (redstone, copper, emerald) give direct measurements;
# mixed-size ores (iron, diamond) are solved via simultaneous equations.
_EXPECTED_BLOCKS: dict[int, float] = {
    3: 0.396,    # emerald — tiny blob, often 0-1 blocks
    4: 2.613,    # diamond_small, iron_small
    7: 4.417,    # lapis
    8: 4.714,    # diamond_medium, diamond_buried, redstone
    9: 5.909,    # iron_middle, gold, iron_upper
    10: 8.649,   # copper
    12: 11.5,    # diamond_large (extrapolated — low-weight feature)
    17: 19.917,  # coal
    20: 14.43,   # copper_large (back-solved from dripstone caves data)
}


def _expected_blocks(size: int) -> float:
    """Expected ore blocks from MC vein algorithm for a given ``size``.

    The MC ``size`` parameter controls the capsule-blob geometry, not the
    block count directly.  This lookup table was derived from 242 real MC
    1.21 chunk observations (see module docstring for derivation).

    For sizes not in the table, linearly interpolates between known values.
    """
    if size in _EXPECTED_BLOCKS:
        return _EXPECTED_BLOCKS[size]
    # Interpolate for unknown sizes
    known = sorted(_EXPECTED_BLOCKS.items())
    if size <= 0:
        return 0.0
    if size < known[0][0]:
        # Below smallest known — linear extrapolation from (0, 0)
        return known[0][1] * size / known[0][0]
    if size > known[-1][0]:
        # Above largest known — power-law extrapolation
        return known[-1][1] * (size / known[-1][0]) ** 1.5
    for i in range(len(known) - 1):
        if known[i][0] <= size <= known[i + 1][0]:
            t = (size - known[i][0]) / (known[i + 1][0] - known[i][0])
            return known[i][1] + t * (known[i + 1][1] - known[i][1])
    return float(size) * 0.7  # fallback

# ---------------------------------------------------------------------------
# Fallback terrain profiles (biome-averaged, from 242 chunks)
# ---------------------------------------------------------------------------

# Replaceable fraction at sampled Y-levels (biome-averaged fallback).
# Column 0 = Y-index (0 = MC Y=-64), column 1 = P(replaceable).
_REPL_PROFILE_SAMPLES = np.array([
    (  0, 0.0000),  # MC Y=-64 (bedrock)
    (  5, 0.5000),  # MC Y=-59 (bedrock → deepslate transition)
    ( 10, 0.6600),  # MC Y=-54
    ( 20, 0.6310),  # MC Y=-44
    ( 30, 0.6500),  # MC Y=-34
    ( 40, 0.6280),  # MC Y=-24
    ( 50, 0.6400),  # MC Y=-14
    ( 60, 0.6340),  # MC Y=-4
    ( 70, 0.6100),  # MC Y=6
    ( 80, 0.5880),  # MC Y=16
    ( 90, 0.5700),  # MC Y=26
    (100, 0.6000),  # MC Y=36
    (110, 0.5100),  # MC Y=46
    (120, 0.5220),  # MC Y=56
    (130, 0.3900),  # MC Y=66
    (140, 0.2150),  # MC Y=76
    (150, 0.1300),  # MC Y=86
    (160, 0.0910),  # MC Y=96
    (170, 0.0480),  # MC Y=106
    (180, 0.0370),  # MC Y=116
    (190, 0.0100),  # MC Y=126
    (200, 0.0040),  # MC Y=136
    (210, 0.0001),  # MC Y=146
    (220, 0.0000),  # MC Y=156
], dtype=np.float64)

# Air adjacency at sampled Y-levels: P(any neighbour is air | solid, y).
_AIR_ADJ_PROFILE_SAMPLES = np.array([
    (  0, 0.005),  # MC Y=-64
    ( 10, 0.025),  # MC Y=-54
    ( 20, 0.036),  # MC Y=-44
    ( 30, 0.042),  # MC Y=-34
    ( 40, 0.043),  # MC Y=-24
    ( 50, 0.046),  # MC Y=-14
    ( 60, 0.042),  # MC Y=-4
    ( 70, 0.051),  # MC Y=6
    ( 80, 0.055),  # MC Y=16
    ( 90, 0.058),  # MC Y=26
    (100, 0.060),  # MC Y=36
    (110, 0.070),  # MC Y=46
    (120, 0.070),  # MC Y=56
    (130, 0.120),  # MC Y=66
    (140, 0.220),  # MC Y=76
    (150, 0.180),  # MC Y=86
    (160, 0.218),  # MC Y=96
    (170, 0.212),  # MC Y=106
    (180, 0.230),  # MC Y=116
    (190, 0.270),  # MC Y=126
    (200, 0.400),  # MC Y=136
    (210, 0.500),  # MC Y=146
    (220, 0.500),  # MC Y=156
], dtype=np.float64)

# Solid fraction samples (kept for legacy code path and _solid_frac_at).
_SOLID_PROFILE_SAMPLES = np.array([
    (  0, 0.8683),  # MC Y=-64
    ( 10, 0.8442),  # MC Y=-54
    ( 20, 0.7927),  # MC Y=-44
    ( 30, 0.7725),  # MC Y=-34
    ( 40, 0.7854),  # MC Y=-24
    ( 50, 0.7900),  # MC Y=-14
    ( 60, 0.8019),  # MC Y=-4
    ( 70, 0.7738),  # MC Y=6
    ( 80, 0.7488),  # MC Y=16
    ( 90, 0.7666),  # MC Y=26
    (100, 0.7579),  # MC Y=36
    (110, 0.6445),  # MC Y=46
    (120, 0.5828),  # MC Y=56
    (130, 0.4544),  # MC Y=66
    (140, 0.2620),  # MC Y=76
    (150, 0.1625),  # MC Y=86
    (160, 0.1020),  # MC Y=96
    (170, 0.0619),  # MC Y=106
    (180, 0.0367),  # MC Y=116
    (190, 0.0137),  # MC Y=126
    (200, 0.0028),  # MC Y=136
    (210, 0.0001),  # MC Y=146
    (220, 0.0000),  # MC Y=156
], dtype=np.float64)


def _build_default_solid_profile() -> np.ndarray:
    """Linearly interpolate the biome-averaged solid fraction to 384 levels.

    Returns shape ``(384, NUM_BIOME_TYPES)`` with the same values for all
    biomes (biome-averaged fallback).
    """
    y_indices = _SOLID_PROFILE_SAMPLES[:, 0]
    fractions = _SOLID_PROFILE_SAMPLES[:, 1]
    all_y = np.arange(_MC_WORLD_HEIGHT, dtype=np.float64)
    profile_1d = np.interp(all_y, y_indices, fractions, right=0.0)
    return np.tile(profile_1d[:, None], (1, NUM_BIOME_TYPES))


def _build_default_replaceable_profile() -> np.ndarray:
    """Linearly interpolate the biome-averaged replaceable fraction.

    Returns shape ``(384, NUM_BIOME_TYPES)`` — biome-averaged fallback.
    """
    y_indices = _REPL_PROFILE_SAMPLES[:, 0]
    fractions = _REPL_PROFILE_SAMPLES[:, 1]
    all_y = np.arange(_MC_WORLD_HEIGHT, dtype=np.float64)
    profile_1d = np.interp(all_y, y_indices, fractions, right=0.0)
    return np.tile(profile_1d[:, None], (1, NUM_BIOME_TYPES))


def _build_default_air_adj_profile() -> np.ndarray:
    """Linearly interpolate the biome-averaged air adjacency fraction.

    Returns shape ``(384, NUM_BIOME_TYPES)`` — biome-averaged fallback.
    """
    y_indices = _AIR_ADJ_PROFILE_SAMPLES[:, 0]
    fractions = _AIR_ADJ_PROFILE_SAMPLES[:, 1]
    all_y = np.arange(_MC_WORLD_HEIGHT, dtype=np.float64)
    profile_1d = np.interp(all_y, y_indices, fractions, right=0.0)
    return np.tile(profile_1d[:, None], (1, NUM_BIOME_TYPES))


class AnalyticalPrior:
    """Precomputed P(ore_type | y_sim, biome) lookup table.

    By default, builds the table from parsed vanilla 1.21.11 worldgen JSON,
    corrected by the empirical replaceable fraction profile and vein geometry.

    Parameters
    ----------
    world_height:
        Height of the simulation world in blocks.
    chunk_volume:
        Volume of a 16x16 chunk column (used for probability normalization
        in legacy mode).  Default assumes 16x16xworld_height.
    solid_profile_path:
        Path to a ``.npz`` file containing per-(y, biome) terrain profiles
        (``solid_fraction``, ``replaceable_fraction``, ``air_adjacency``).
        Falls back to hardcoded biome-averaged interpolation if not found.
    apply_solid_correction:
        If False, skip the terrain correction (original behavior).
        Useful for A/B comparison.
    calibration_path:
        Path to a ``.npz`` file containing per-ore per-Y-band calibration
        factors.  Generated by ``prospect_rl.tools.calibrate_prior``.
        Falls back gracefully (no correction) if the file is not found.
    apply_calibration:
        If False, skip the empirical calibration step.
        Useful for A/B comparison.
    """

    def __init__(
        self,
        world_height: int = 64,
        chunk_volume: int | None = None,
        solid_profile_path: str | Path | None = "data/solid_fraction_profile.npz",
        apply_solid_correction: bool = True,
        calibration_path: str | Path | None = "data/calibration_table.npz",
        apply_calibration: bool = False,
    ) -> None:
        self._world_height = world_height
        self._chunk_volume = chunk_volume or (16 * 16 * world_height)

        # Load terrain profiles before building the table
        profiles = self._load_profiles(
            solid_profile_path, apply_solid_correction,
        )
        self._solid_profile = profiles["solid"]
        self._replaceable_profile = profiles["replaceable"]
        self._air_adj_profile = profiles["air_adj"]

        # Lookup table: (ore_type, y_sim, biome) -> probability
        self._table = np.zeros(
            (NUM_ORE_TYPES, world_height, NUM_BIOME_TYPES),
            dtype=np.float64,
        )
        self._build_table()

        # Apply empirical calibration (multiplicative correction per Y-band)
        self._calibration = self._load_calibration(
            calibration_path, apply_calibration,
        )
        if self._calibration is not None:
            self._apply_calibration()

        # Per-ore cluster radii (from vein sizes)
        self._cluster_radii: dict[int, float] = {}
        for otc in ORE_TYPE_CONFIGS:
            self._cluster_radii[otc.ore_index] = max(
                1.0, math.sqrt(otc.typical_vein_size),
            )

    def _load_profiles(
        self,
        path: str | Path | None,
        apply: bool,
    ) -> dict[str, np.ndarray]:
        """Load per-(y_mc, biome) terrain profiles.

        Returns dict with keys ``solid``, ``replaceable``, ``air_adj``,
        each shape ``(384, NUM_BIOME_TYPES)``.  Index 0 = MC Y=-64.
        """
        ones = np.ones((_MC_WORLD_HEIGHT, NUM_BIOME_TYPES), dtype=np.float64)
        zeros = np.zeros(
            (_MC_WORLD_HEIGHT, NUM_BIOME_TYPES), dtype=np.float64,
        )

        if not apply:
            return {
                "solid": ones.copy(),
                "replaceable": ones.copy(),
                "air_adj": zeros,
            }

        solid = None
        replaceable = None
        air_adj = None

        if path is not None:
            try:
                p = Path(path)
                if p.exists():
                    data = np.load(p, allow_pickle=False)

                    # Solid fraction (always present)
                    if "solid_fraction" in data:
                        sf = data["solid_fraction"]
                        if sf.shape == (_MC_WORLD_HEIGHT, NUM_BIOME_TYPES):
                            solid = sf.astype(np.float64)

                    # Replaceable fraction (new profile)
                    if "replaceable_fraction" in data:
                        rf = data["replaceable_fraction"]
                        if rf.shape == (_MC_WORLD_HEIGHT, NUM_BIOME_TYPES):
                            replaceable = rf.astype(np.float64)

                    # Air adjacency (new profile)
                    if "air_adjacency" in data:
                        aa = data["air_adjacency"]
                        if aa.shape == (_MC_WORLD_HEIGHT, NUM_BIOME_TYPES):
                            air_adj = aa.astype(np.float64)

            except (FileNotFoundError, KeyError, ValueError):
                pass

        # Fill in missing profiles with fallbacks
        if solid is None:
            solid = _build_default_solid_profile()
        if replaceable is None:
            # If .npz doesn't have replaceable, fall back to solid * 0.98
            # (rough approximation — most solid underground is stone)
            if solid is not None:
                replaceable = _build_default_replaceable_profile()
            else:
                replaceable = ones.copy()
        if air_adj is None:
            air_adj = _build_default_air_adj_profile()

        return {
            "solid": solid,
            "replaceable": replaceable,
            "air_adj": air_adj,
        }

    def _solid_frac_at(self, mc_y: int, biome_id: int) -> float:
        """Look up P(solid | mc_y, biome) from the profile."""
        profile_y = max(0, min(mc_y - MC_Y_MIN, _MC_WORLD_HEIGHT - 1))
        biome_id = max(0, min(biome_id, NUM_BIOME_TYPES - 1))
        return float(self._solid_profile[profile_y, biome_id])

    def _replaceable_frac_at(self, mc_y: int, biome_id: int) -> float:
        """Look up P(replaceable | mc_y, biome) from the profile."""
        profile_y = max(0, min(mc_y - MC_Y_MIN, _MC_WORLD_HEIGHT - 1))
        biome_id = max(0, min(biome_id, NUM_BIOME_TYPES - 1))
        return float(self._replaceable_profile[profile_y, biome_id])

    def _air_adj_at(self, mc_y: int, biome_id: int) -> float:
        """Look up P(any neighbour is air | solid, mc_y, biome)."""
        profile_y = max(0, min(mc_y - MC_Y_MIN, _MC_WORLD_HEIGHT - 1))
        biome_id = max(0, min(biome_id, NUM_BIOME_TYPES - 1))
        return float(self._air_adj_profile[profile_y, biome_id])

    # ------------------------------------------------------------------
    # Empirical calibration
    # ------------------------------------------------------------------

    def _load_calibration(
        self,
        path: str | Path | None,
        apply: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Load per-ore per-Y-band calibration factors.

        Returns ``(factors, band_centers)`` or ``None`` if unavailable.
        ``factors`` has shape ``(NUM_ORE_TYPES, n_bands, NUM_BIOME_TYPES)``,
        ``band_centers`` has shape ``(n_bands,)`` (MC Y-index, 0 = MC Y=-64).
        """
        if not apply or path is None:
            return None

        try:
            p = Path(path)
            if not p.exists():
                return None
            data = np.load(p, allow_pickle=False)
            factors = data["calibration_factors"]  # (8, n_bands, 5)
            band_centers = data["band_centers"]  # (n_bands,)
            if (
                factors.ndim == 3
                and factors.shape[0] == NUM_ORE_TYPES
                and factors.shape[2] == NUM_BIOME_TYPES
                and band_centers.ndim == 1
                and band_centers.shape[0] == factors.shape[1]
            ):
                return (
                    factors.astype(np.float64),
                    band_centers.astype(np.float64),
                )
        except (FileNotFoundError, KeyError, ValueError):
            pass

        return None

    def _apply_calibration(self) -> None:
        """Multiply the prior table by smoothly-interpolated calibration factors.

        For each ``(ore, y_sim, biome)``, maps ``y_sim`` to MC Y-index,
        linearly interpolates the per-band factor, and multiplies in place.
        """
        if self._calibration is None:
            return

        factors, band_centers = self._calibration
        h = self._world_height

        for ore_idx in range(NUM_ORE_TYPES):
            for biome_id in range(NUM_BIOME_TYPES):
                # Build a 1-D calibration curve for this (ore, biome).
                curve = factors[ore_idx, :, biome_id]

                # Interpolate to every sim Y-level.
                for y_sim in range(h):
                    mc_y_idx = (y_sim / h) * _MC_WORLD_HEIGHT
                    factor = float(np.interp(mc_y_idx, band_centers, curve))
                    self._table[ore_idx, y_sim, biome_id] *= factor

        np.clip(self._table, 0.0, 1.0, out=self._table)

    @staticmethod
    def _compute_effective_pmf(
        height_dist: object,
        world_min_y: int,
        world_max_y: int,
    ) -> dict[int, float]:
        """Compute height PMF preserving mass loss for out-of-world Y levels.

        Unlike ``compute_height_pmf`` (which renormalizes to sum=1.0 after
        clamping to world bounds), this preserves the original probability
        mass so that attempts targeting Y outside the world are correctly
        treated as wasted.

        Returns ``{mc_y: probability}`` where the sum is <= 1.0.
        """
        from env.worldgen_parser import compute_height_pmf

        # Compute over a very wide range to avoid clamping
        full_pmf = compute_height_pmf(
            height_dist, world_min_y=-1000, world_max_y=1000,
        )
        # Filter to in-world Y levels only (out-of-world mass is lost)
        return {
            y: p for y, p in full_pmf.items()
            if world_min_y <= y <= world_max_y
        }

    def _build_table(self) -> None:
        """Compute per-voxel ore probability from parsed worldgen JSON.

        For each biome group, iterates over the placed features active in
        that biome and accumulates per-voxel probability using:

        .. math::

            p_{voxel} = E[attempts] \\times P_{eff}(y) \\times
                        expected\\_blocks(size) / 256 \\times
                        repl\\_frac(y, biome)
                        \\times (1 - discard \\times P_{adj\\_air}(y, biome))

        where ``P_eff(y)`` preserves mass loss for out-of-world Y levels,
        ``expected_blocks(size)`` maps MC's ``size`` parameter to the actual
        average block count placed by the vein algorithm, ``repl_frac`` is
        the fraction of blocks in ``stone_ore_replaceables``, and the
        air-adjacency term models the ``discard_chance_on_air_exposure``
        parameter.

        Falls back to legacy ORE_SPAWN_CONFIGS-based computation if the
        worldgen parser is unavailable.
        """
        try:
            from env.worldgen_parser import (
                get_ore_features_per_biome_group,
                load_worldgen,
            )
            wg = load_worldgen()
        except (FileNotFoundError, ImportError):
            self._build_table_legacy()
            return

        h = self._world_height

        # Build biome group -> set of placed feature names
        biome_group_features = get_ore_features_per_biome_group(
            wg.biome_profiles,
        )

        for biome_group_id in range(NUM_BIOME_TYPES):
            active_features = biome_group_features.get(biome_group_id, set())

            for feat_name in active_features:
                placed = wg.placed_features.get(feat_name)
                if placed is None or placed.configured is None:
                    continue

                cfg = placed.configured

                # Map to our 8 ore types
                ore_idx = _MC_BLOCK_TO_ORE_INDEX.get(cfg.ore_block)
                if ore_idx is None:
                    continue

                # Expected attempts per chunk
                e_attempts = placed.attempt_model.expected_attempts

                # Height PMF preserving out-of-world mass loss
                pmf = self._compute_effective_pmf(
                    placed.height_distribution, wg.min_y, wg.max_y,
                )

                # Expected blocks from vein geometry (not raw size)
                geo_blocks = _expected_blocks(cfg.size)

                # Accumulate contribution to table
                for mc_y, p_y in pmf.items():
                    sim_y = int((mc_y - MC_Y_MIN) / MC_Y_RANGE * h)
                    if 0 <= sim_y < h:
                        # Replaceable fraction at this Y and biome
                        rf = self._replaceable_frac_at(
                            mc_y, biome_group_id,
                        )

                        # Air-exposure discard correction
                        p_air = self._air_adj_at(
                            mc_y, biome_group_id,
                        )
                        keep_frac = max(
                            0.0,
                            1.0 - cfg.discard_chance_on_air_exposure * p_air,
                        )

                        eff_blocks = geo_blocks * keep_frac

                        p_voxel = (
                            e_attempts * p_y * eff_blocks / 256.0
                        ) * rf

                        self._table[
                            ore_idx, sim_y, biome_group_id
                        ] += p_voxel

        # Clamp to [0, 1]
        np.clip(self._table, 0.0, 1.0, out=self._table)

    def _build_table_legacy(self) -> None:
        """Fallback: build table from ORE_SPAWN_CONFIGS (pre-JSON path)."""
        configs_by_ore: dict[int, list[OreSpawnConfig]] = {}
        for cfg in ORE_SPAWN_CONFIGS:
            bt = int(cfg.block_type)
            for i, ore_bt in enumerate(ORE_TYPES):
                if int(ore_bt) == bt:
                    configs_by_ore.setdefault(i, []).append(cfg)
                    break

        for ore_idx, cfgs in configs_by_ore.items():
            for cfg in cfgs:
                self._add_config_contribution_legacy(ore_idx, cfg)

        np.clip(self._table, 0.0, 1.0, out=self._table)

    def _add_config_contribution_legacy(
        self, ore_idx: int, cfg: OreSpawnConfig,
    ) -> None:
        """Add a single spawn config's contribution (legacy path)."""
        h = self._world_height

        sim_y_min = max(0, int((cfg.y_min_mc - MC_Y_MIN) / MC_Y_RANGE * h))
        sim_y_max = min(
            h - 1, int((cfg.y_max_mc - MC_Y_MIN) / MC_Y_RANGE * h),
        )

        if sim_y_min > sim_y_max:
            return

        base_prob = (cfg.spawn_tries * cfg.spawn_size) / self._chunk_volume
        effective_prob = base_prob * (1.0 - cfg.cluster_threshold * 0.5)

        for y_sim in range(sim_y_min, sim_y_max + 1):
            if cfg.distribution == "triangle":
                peak_mc = cfg.peak_mc
                if peak_mc is None:
                    peak_mc = (cfg.y_min_mc + cfg.y_max_mc) // 2
                mc_y = MC_Y_MIN + (y_sim / h) * MC_Y_RANGE
                half_range = (cfg.y_max_mc - cfg.y_min_mc) / 2.0
                if half_range > 0:
                    dist_from_peak = abs(mc_y - peak_mc)
                    weight = max(0.0, 1.0 - dist_from_peak / half_range)
                else:
                    weight = 1.0
            else:
                weight = 1.0

            prob = effective_prob * weight

            # Map sim Y back to MC Y for solid profile lookup
            mc_y_lookup = int(MC_Y_MIN + (y_sim / h) * MC_Y_RANGE)

            if cfg.biomes is not None:
                for biome_id in cfg.biomes:
                    solid_frac = self._solid_frac_at(
                        mc_y_lookup, int(biome_id),
                    )
                    self._table[
                        ore_idx, y_sim, int(biome_id)
                    ] += prob * solid_frac
            else:
                for biome_id in range(NUM_BIOME_TYPES):
                    solid_frac = self._solid_frac_at(
                        mc_y_lookup, biome_id,
                    )
                    self._table[ore_idx, y_sim, biome_id] += prob * solid_frac

    def query(self, y_sim: int, biome_id: int) -> np.ndarray:
        """Return P(ore_type) at a specific position.

        Returns shape ``(8,)`` — probability per ore type.
        """
        y_sim = max(0, min(y_sim, self._world_height - 1))
        biome_id = max(0, min(biome_id, NUM_BIOME_TYPES - 1))
        return self._table[:, y_sim, biome_id].copy().astype(np.float32)

    def query_chunk(
        self,
        cx: int,
        cz: int,
        biome_map: np.ndarray,
        chunk_size_xz: int = 16,
    ) -> np.ndarray:
        """Return expected ore count per type in a chunk column.

        Integrates over all Y levels and the chunk's XZ area using
        the biome map to look up biome-specific probabilities.

        Returns shape ``(8,)`` — expected number of each ore type.
        """
        expected = np.zeros(NUM_ORE_TYPES, dtype=np.float64)

        x0, x1 = cx * chunk_size_xz, min(
            (cx + 1) * chunk_size_xz, biome_map.shape[0],
        )
        z0, z1 = cz * chunk_size_xz, min(
            (cz + 1) * chunk_size_xz, biome_map.shape[1],
        )

        # Collect unique biomes in this chunk
        biome_counts: dict[int, int] = {}
        for x in range(x0, x1):
            for z in range(z0, z1):
                b = int(biome_map[x, z])
                biome_counts[b] = biome_counts.get(b, 0) + 1

        # Sum probabilities over Y levels, weighted by biome coverage
        for biome_id, count in biome_counts.items():
            biome_id = max(0, min(biome_id, NUM_BIOME_TYPES - 1))
            for y in range(self._world_height):
                expected += self._table[:, y, biome_id] * count

        return expected.astype(np.float32)

    def get_cluster_radius(self, ore_type: int) -> float:
        """Return the cluster propagation radius for an ore type."""
        return self._cluster_radii.get(ore_type, 4.0)
