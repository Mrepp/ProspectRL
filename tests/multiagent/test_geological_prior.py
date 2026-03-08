"""Tests for terrain-corrected AnalyticalPrior.

Covers:
- Replaceable fraction, air adjacency, and solid fraction profile corrections
- Expected blocks lookup table
- Effective height PMF with mass loss
- Calibration loading/application
- Profile loading and fallback logic
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from prospect_rl.config import NUM_BIOME_TYPES, NUM_ORE_TYPES


@pytest.fixture()
def prior_corrected():
    """Prior with terrain corrections (default)."""
    from multiagent.geological_prior import AnalyticalPrior

    return AnalyticalPrior(world_height=384)


@pytest.fixture()
def prior_uncorrected():
    """Prior without terrain corrections (original behavior)."""
    from multiagent.geological_prior import AnalyticalPrior

    return AnalyticalPrior(world_height=384, apply_solid_correction=False)


class TestTerrainCorrection:
    """Tests for the terrain correction logic."""

    def test_correction_reduces_predictions(
        self, prior_corrected, prior_uncorrected,
    ):
        """Corrected prior should have lower total predictions than uncorrected."""
        total_corrected = prior_corrected._table.sum()
        total_uncorrected = prior_uncorrected._table.sum()
        assert total_corrected < total_uncorrected
        # Should reduce significantly (replaceable < solid < 1.0)
        assert total_corrected < total_uncorrected * 0.5

    def test_table_shape(self, prior_corrected):
        """Table should have expected shape."""
        assert prior_corrected._table.shape == (NUM_ORE_TYPES, 384, NUM_BIOME_TYPES)

    def test_table_non_negative(self, prior_corrected):
        """All probabilities should be non-negative."""
        assert (prior_corrected._table >= 0).all()

    def test_table_at_most_one(self, prior_corrected):
        """All probabilities should be at most 1.0."""
        assert (prior_corrected._table <= 1.0).all()

    def test_deep_ores_less_affected(
        self, prior_corrected, prior_uncorrected,
    ):
        """Deep ores (diamond, idx=3) should be less reduced than shallow ores (coal, idx=0).

        Diamond spawns below y=80 (MC Y=16) where replaceable fraction is ~65%.
        Coal spans up to y=192 (MC Y=128) where replaceable fraction drops to ~5%.
        """
        coal_idx = 0
        diamond_idx = 3

        coal_corrected = prior_corrected._table[coal_idx].sum()
        coal_uncorrected = prior_uncorrected._table[coal_idx].sum()
        diamond_corrected = prior_corrected._table[diamond_idx].sum()
        diamond_uncorrected = prior_uncorrected._table[diamond_idx].sum()

        coal_ratio = coal_corrected / max(coal_uncorrected, 1e-12)
        diamond_ratio = diamond_corrected / max(diamond_uncorrected, 1e-12)

        # Diamond should retain a higher fraction than coal
        assert diamond_ratio > coal_ratio

    def test_high_y_levels_zeroed(self, prior_corrected):
        """Above surface (y>220, MC Y>156), predictions should be near zero."""
        high_y_total = prior_corrected._table[:, 220:, :].sum()
        assert high_y_total < 1e-10

    def test_query_returns_correct_shape(self, prior_corrected):
        """query() should return shape (8,) float32."""
        result = prior_corrected.query(100, 0)
        assert result.shape == (8,)
        assert result.dtype == np.float32

    def test_query_chunk_returns_correct_shape(self, prior_corrected):
        """query_chunk() should return shape (8,) float32."""
        biome_map = np.zeros((64, 64), dtype=np.int8)
        result = prior_corrected.query_chunk(0, 0, biome_map)
        assert result.shape == (8,)
        assert result.dtype == np.float32


class TestExpectedBlocks:
    """Tests for the _expected_blocks vein geometry lookup."""

    def test_known_sizes_positive(self):
        """All known sizes should return positive block counts."""
        from multiagent.geological_prior import _expected_blocks

        for size in [3, 4, 7, 8, 9, 10, 12, 17, 20]:
            assert _expected_blocks(size) > 0, f"size={size} should be positive"

    def test_generally_increasing(self):
        """Larger sizes should generally produce more blocks (non-strict).

        Note: _expected_blocks values are empirically derived, so they are not
        strictly monotonic.  For example, ore_copper_large (size=20) produces
        fewer blocks than ore_coal (size=17) due to different biome/geometry
        characteristics.  We only check that the general trend is upward.
        """
        from multiagent.geological_prior import _expected_blocks

        # Check that small sizes produce fewer blocks than large sizes
        assert _expected_blocks(3) < _expected_blocks(10)
        assert _expected_blocks(4) < _expected_blocks(12)
        assert _expected_blocks(7) < _expected_blocks(17)

    def test_small_sizes_less_than_parameter(self):
        """For small sizes, expected blocks should be less than the size parameter."""
        from multiagent.geological_prior import _expected_blocks

        for size in [3, 4, 7, 8]:
            blocks = _expected_blocks(size)
            assert blocks < size, (
                f"size={size}: expected {blocks:.3f} blocks should be < {size}"
            )

    def test_size_zero_returns_zero(self):
        """Size 0 should return 0 blocks."""
        from multiagent.geological_prior import _expected_blocks

        assert _expected_blocks(0) == 0.0

    def test_interpolation_for_unknown_size(self):
        """Unknown sizes should interpolate between known values."""
        from multiagent.geological_prior import _expected_blocks

        # Size 5 should be between size 4 and size 7
        assert _expected_blocks(4) < _expected_blocks(5) < _expected_blocks(7)


class TestEffectivePMF:
    """Tests for _compute_effective_pmf mass preservation."""

    def test_in_world_distribution_sums_near_one(self):
        """A distribution entirely within world bounds should sum to ~1.0."""
        from env.worldgen_parser import HeightDistribution
        from multiagent.geological_prior import AnalyticalPrior

        dist = HeightDistribution(
            dist_type="uniform",
            min_inclusive=0,
            max_inclusive=100,
        )
        pmf = AnalyticalPrior._compute_effective_pmf(dist, -64, 320)
        total_mass = sum(pmf.values())
        assert 0.99 < total_mass <= 1.0, (
            f"In-world distribution should sum to ~1.0, got {total_mass:.4f}"
        )

    def test_out_of_world_distribution_loses_mass(self):
        """A distribution extending below Y=-64 should sum to < 1.0."""
        from env.worldgen_parser import HeightDistribution
        from multiagent.geological_prior import AnalyticalPrior

        # Diamond-like: trapezoid from Y=-144 to Y=16 (triangle, plateau=0)
        dist = HeightDistribution(
            dist_type="trapezoid",
            min_inclusive=-144,
            max_inclusive=16,
            plateau=0,
        )
        pmf = AnalyticalPrior._compute_effective_pmf(dist, -64, 320)
        total_mass = sum(pmf.values())
        assert total_mass < 0.95, (
            f"Out-of-world distribution should lose mass, got {total_mass:.4f}"
        )

    def test_all_keys_within_world(self):
        """All PMF keys should be within world bounds."""
        from env.worldgen_parser import HeightDistribution
        from multiagent.geological_prior import AnalyticalPrior

        dist = HeightDistribution(
            dist_type="trapezoid",
            min_inclusive=-200,
            max_inclusive=400,
            plateau=0,
        )
        pmf = AnalyticalPrior._compute_effective_pmf(dist, -64, 320)
        for y in pmf:
            assert -64 <= y <= 320, f"Y={y} is outside world bounds [-64, 320]"


class TestProfileLoading:
    """Tests for terrain profile loading and fallback logic."""

    def test_fallback_on_missing_file(self):
        """Should fall back to interpolated profiles when .npz is missing."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=384,
            solid_profile_path="/nonexistent/path/profile.npz",
        )
        assert prior._table.shape == (NUM_ORE_TYPES, 384, NUM_BIOME_TYPES)
        assert prior._table.sum() > 0

    def test_fallback_on_none_path(self):
        """Should fall back to interpolated profiles when path is None."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=384,
            solid_profile_path=None,
        )
        assert prior._table.shape == (NUM_ORE_TYPES, 384, NUM_BIOME_TYPES)
        assert prior._table.sum() > 0

    def test_solid_profile_shape(self):
        """The internal solid profile should have correct shape."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(world_height=384)
        assert prior._solid_profile.shape == (384, NUM_BIOME_TYPES)

    def test_replaceable_profile_shape(self):
        """The internal replaceable profile should have correct shape."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(world_height=384)
        assert prior._replaceable_profile.shape == (384, NUM_BIOME_TYPES)

    def test_air_adj_profile_shape(self):
        """The internal air adjacency profile should have correct shape."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(world_height=384)
        assert prior._air_adj_profile.shape == (384, NUM_BIOME_TYPES)

    def test_solid_profile_range(self):
        """Solid profile values should be in [0, 1]."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(world_height=384)
        assert (prior._solid_profile >= 0).all()
        assert (prior._solid_profile <= 1).all()

    def test_replaceable_profile_range(self):
        """Replaceable profile values should be in [0, 1]."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(world_height=384)
        assert (prior._replaceable_profile >= 0).all()
        assert (prior._replaceable_profile <= 1).all()

    def test_air_adj_profile_range(self):
        """Air adjacency profile values should be in [0, 1]."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(world_height=384)
        assert (prior._air_adj_profile >= 0).all()
        assert (prior._air_adj_profile <= 1).all()

    def test_replaceable_less_than_solid(self):
        """Replaceable fraction should be <= solid fraction (replaceable is a subset)."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(world_height=384)
        # Allow small tolerance for interpolation artifacts
        assert (prior._replaceable_profile <= prior._solid_profile + 0.01).all()

    def test_no_correction_gives_all_ones_solid(self):
        """apply_solid_correction=False should use an all-ones solid profile."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=384,
            apply_solid_correction=False,
        )
        np.testing.assert_array_equal(
            prior._solid_profile,
            np.ones((384, NUM_BIOME_TYPES)),
        )

    def test_no_correction_gives_all_ones_replaceable(self):
        """apply_solid_correction=False should use all-ones replaceable profile."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=384,
            apply_solid_correction=False,
        )
        np.testing.assert_array_equal(
            prior._replaceable_profile,
            np.ones((384, NUM_BIOME_TYPES)),
        )

    def test_no_correction_gives_zero_air_adj(self):
        """apply_solid_correction=False should use all-zeros air adjacency."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=384,
            apply_solid_correction=False,
        )
        np.testing.assert_array_equal(
            prior._air_adj_profile,
            np.zeros((384, NUM_BIOME_TYPES)),
        )

    def test_npz_profile_loads_all_three(self):
        """Should load solid, replaceable, and air_adjacency from .npz."""
        from multiagent.geological_prior import AnalyticalPrior

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            fake_solid = np.full((384, NUM_BIOME_TYPES), 0.8, dtype=np.float64)
            fake_repl = np.full((384, NUM_BIOME_TYPES), 0.5, dtype=np.float64)
            fake_air = np.full((384, NUM_BIOME_TYPES), 0.1, dtype=np.float64)
            np.savez(
                f.name,
                solid_fraction=fake_solid,
                replaceable_fraction=fake_repl,
                air_adjacency=fake_air,
            )
            tmp_path = f.name

        try:
            prior = AnalyticalPrior(
                world_height=384,
                solid_profile_path=tmp_path,
            )
            np.testing.assert_array_almost_equal(
                prior._solid_profile, fake_solid,
            )
            np.testing.assert_array_almost_equal(
                prior._replaceable_profile, fake_repl,
            )
            np.testing.assert_array_almost_equal(
                prior._air_adj_profile, fake_air,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_npz_with_only_solid_uses_fallback_for_others(self):
        """If .npz only has solid_fraction, replaceable and air_adj use fallbacks."""
        from multiagent.geological_prior import AnalyticalPrior

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            fake_solid = np.full((384, NUM_BIOME_TYPES), 0.7, dtype=np.float64)
            np.savez(f.name, solid_fraction=fake_solid)
            tmp_path = f.name

        try:
            prior = AnalyticalPrior(
                world_height=384,
                solid_profile_path=tmp_path,
            )
            np.testing.assert_array_almost_equal(
                prior._solid_profile, fake_solid,
            )
            # Replaceable and air_adj should use fallback (not all-ones/zeros)
            assert prior._replaceable_profile.sum() > 0
            assert prior._air_adj_profile.sum() > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestDefaultProfiles:
    """Tests for the hardcoded fallback profiles."""

    def test_solid_interpolation_monotonic_above_surface(self):
        """Solid fraction should generally decrease above surface level."""
        from multiagent.geological_prior import _build_default_solid_profile

        profile = _build_default_solid_profile()
        above_surface = profile[100:220, 0]
        assert above_surface[0] > above_surface[-1]

    def test_solid_interpolation_zero_above_220(self):
        """Solid profile should be zero above Y=220 (MC Y=156)."""
        from multiagent.geological_prior import _build_default_solid_profile

        profile = _build_default_solid_profile()
        assert (profile[220:, :] == 0.0).all()

    def test_solid_interpolation_positive_at_bedrock(self):
        """Solid profile should be ~0.87 at bedrock level."""
        from multiagent.geological_prior import _build_default_solid_profile

        profile = _build_default_solid_profile()
        assert profile[0, 0] > 0.8

    def test_replaceable_zero_at_bedrock(self):
        """Replaceable fraction should be ~0 at bedrock level (Y=-64)."""
        from multiagent.geological_prior import _build_default_replaceable_profile

        profile = _build_default_replaceable_profile()
        assert profile[0, 0] < 0.01

    def test_replaceable_positive_underground(self):
        """Replaceable fraction should be substantial at deepslate levels."""
        from multiagent.geological_prior import _build_default_replaceable_profile

        profile = _build_default_replaceable_profile()
        # At Y=20 (MC Y=-44), should be ~0.63
        assert profile[20, 0] > 0.5

    def test_air_adj_low_underground(self):
        """Air adjacency should be low deep underground."""
        from multiagent.geological_prior import _build_default_air_adj_profile

        profile = _build_default_air_adj_profile()
        # At Y=20 (MC Y=-44), P(adj_air) should be < 5%
        assert profile[20, 0] < 0.05

    def test_air_adj_higher_near_surface(self):
        """Air adjacency should be higher near surface than deep underground."""
        from multiagent.geological_prior import _build_default_air_adj_profile

        profile = _build_default_air_adj_profile()
        deep = profile[20, 0]  # MC Y=-44
        surface = profile[130, 0]  # MC Y=66
        assert surface > deep


class TestCalibration:
    """Tests for empirical calibration loading and application."""

    def test_calibration_fallback_on_missing_file(self):
        """Should not crash when calibration file is missing."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=384,
            calibration_path="/nonexistent/path.npz",
            apply_calibration=True,
        )
        assert prior._table.shape == (
            NUM_ORE_TYPES, 384, NUM_BIOME_TYPES,
        )
        assert prior._table.sum() > 0

    def test_calibration_disabled_by_default(self):
        """Calibration should be disabled by default."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(world_height=384)
        assert prior._calibration is None

    def test_calibration_disabled_explicit(self):
        """apply_calibration=False should match missing-file case."""
        from multiagent.geological_prior import AnalyticalPrior

        prior_disabled = AnalyticalPrior(
            world_height=384,
            apply_calibration=False,
        )
        prior_missing = AnalyticalPrior(
            world_height=384,
            calibration_path="/nonexistent/path.npz",
            apply_calibration=True,
        )
        np.testing.assert_array_equal(
            prior_disabled._table, prior_missing._table,
        )

    def test_calibration_preserves_shape_and_bounds(self):
        """Calibrated table should have same shape and valid bounds."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=384,
            apply_calibration=True,
        )
        assert prior._table.shape == (
            NUM_ORE_TYPES, 384, NUM_BIOME_TYPES,
        )
        assert (prior._table >= 0).all()
        assert (prior._table <= 1.0).all()

    def test_synthetic_calibration_halves_table(self):
        """A calibration file with all-0.5 factors should halve values."""
        from multiagent.geological_prior import AnalyticalPrior

        # Get uncalibrated baseline.
        prior_base = AnalyticalPrior(
            world_height=384,
            apply_calibration=False,
        )
        base_table = prior_base._table.copy()

        # Create synthetic calibration with factor=0.5 everywhere.
        n_bands = 16
        factors = np.full(
            (NUM_ORE_TYPES, n_bands, NUM_BIOME_TYPES),
            0.5, dtype=np.float64,
        )
        band_size = 384 // n_bands
        band_edges = np.arange(
            0, 384 + 1, band_size, dtype=np.int32,
        )
        band_edges[-1] = 384
        band_centers = (
            (band_edges[:-1] + band_edges[1:]) / 2.0
        ).astype(np.float64)

        with tempfile.NamedTemporaryFile(
            suffix=".npz", delete=False,
        ) as f:
            np.savez(
                f.name,
                calibration_factors=factors,
                band_centers=band_centers,
                band_edges=band_edges,
                num_chunks=np.array(100),
            )
            tmp_path = f.name

        try:
            prior_cal = AnalyticalPrior(
                world_height=384,
                calibration_path=tmp_path,
                apply_calibration=True,
            )
            # Total should be approximately halved.
            cal_total = prior_cal._table.sum()
            base_total = base_table.sum()
            assert cal_total < base_total * 0.6
            assert cal_total > base_total * 0.4
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_calibration_none_path(self):
        """calibration_path=None should skip calibration."""
        from multiagent.geological_prior import AnalyticalPrior

        prior = AnalyticalPrior(
            world_height=384,
            calibration_path=None,
        )
        assert prior._calibration is None
        assert prior._table.sum() > 0
