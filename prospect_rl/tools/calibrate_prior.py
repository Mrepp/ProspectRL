"""Compute per-ore per-Y-band empirical calibration factors for AnalyticalPrior.

Compares the analytical prior's predictions against empirical ore frequencies
from real Minecraft chunk data and saves multiplicative correction factors
that align the prior to observed rates.

Usage::

    python -m prospect_rl.tools.calibrate_prior \\
        --cache-dir data/chunk_cache/combined \\
        --output data/calibration_table.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from prospect_rl.config import NUM_BIOME_TYPES, NUM_ORE_TYPES
from prospect_rl.tools.evaluate_prior import accumulate_empirical_stats

logger = logging.getLogger(__name__)

_ORE_NAMES = ["coal", "iron", "gold", "diamond", "redstone", "emerald", "lapis", "copper"]

# Number of Y-bands and MC world height constants.
N_BANDS = 16
_MC_WORLD_HEIGHT = 384
_BAND_SIZE = _MC_WORLD_HEIGHT // N_BANDS  # 24 levels per band

# Minimum total voxels in a (band, biome) cell before we trust the estimate.
_MIN_VOXELS = 10_000

# Clamp extreme calibration factors to avoid pathological corrections.
_FACTOR_MIN = 0.01
_FACTOR_MAX = 50.0


def compute_calibration_table(
    cache_dir: str,
    n_bands: int = N_BANDS,
    min_voxels: int = _MIN_VOXELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute calibration factors from empirical chunk data.

    Returns
    -------
    factors : ndarray, shape (NUM_ORE_TYPES, n_bands, NUM_BIOME_TYPES)
        Multiplicative correction: ``calibrated = uncalibrated * factor``.
    band_centers : ndarray, shape (n_bands,)
        MC Y-level at the center of each band.
    band_edges : ndarray, shape (n_bands + 1,)
        MC Y-level boundaries (indices into 0..383 profile).
    num_chunks : int
    """
    from multiagent.geological_prior import AnalyticalPrior

    stats = accumulate_empirical_stats(cache_dir)
    h = stats.world_height
    assert h == _MC_WORLD_HEIGHT, f"Expected world_height={_MC_WORLD_HEIGHT}, got {h}"

    # Build uncalibrated prior at full MC resolution.
    prior = AnalyticalPrior(
        world_height=h,
        apply_calibration=False,
    )

    band_size = h // n_bands
    band_edges = np.arange(0, h + 1, band_size, dtype=np.int32)
    # Ensure last edge covers full range
    band_edges[-1] = h
    band_centers = ((band_edges[:-1] + band_edges[1:]) / 2.0).astype(np.float64)

    factors = np.ones((NUM_ORE_TYPES, n_bands, NUM_BIOME_TYPES), dtype=np.float64)

    for ore_idx in range(NUM_ORE_TYPES):
        for band_idx in range(n_bands):
            y_lo = int(band_edges[band_idx])
            y_hi = int(band_edges[band_idx + 1])

            # Compute biome-averaged factor as fallback.
            total_emp_all = 0.0
            total_vox_all = 0.0
            total_pred_all = 0.0
            for b in range(NUM_BIOME_TYPES):
                total_emp_all += float(stats.ore_count[ore_idx, y_lo:y_hi, b].sum())
                total_vox_all += float(stats.total_voxels[y_lo:y_hi, b].sum())
                total_pred_all += float(prior._table[ore_idx, y_lo:y_hi, b].sum())

            if total_vox_all > 0 and total_pred_all > 0:
                avg_factor = (total_emp_all / total_vox_all) / (total_pred_all / (y_hi - y_lo))
            else:
                avg_factor = 1.0
            avg_factor = float(np.clip(avg_factor, _FACTOR_MIN, _FACTOR_MAX))

            for biome_id in range(NUM_BIOME_TYPES):
                vox = float(stats.total_voxels[y_lo:y_hi, biome_id].sum())
                emp_count = float(stats.ore_count[ore_idx, y_lo:y_hi, biome_id].sum())
                pred_sum = float(prior._table[ore_idx, y_lo:y_hi, biome_id].sum())
                n_y = y_hi - y_lo

                if vox < min_voxels:
                    # Not enough data — use biome-averaged fallback.
                    factors[ore_idx, band_idx, biome_id] = avg_factor
                    continue

                if pred_sum <= 0:
                    # Prior predicts zero — no correction possible.
                    factors[ore_idx, band_idx, biome_id] = 1.0
                    continue

                emp_rate = emp_count / vox
                pred_rate = pred_sum / n_y
                factor = emp_rate / pred_rate
                factors[ore_idx, band_idx, biome_id] = float(
                    np.clip(factor, _FACTOR_MIN, _FACTOR_MAX),
                )

    return factors, band_centers, band_edges, stats.num_chunks


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Compute empirical calibration factors for AnalyticalPrior.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/chunk_cache/combined",
        help="Directory containing .npz cache files",
    )
    parser.add_argument(
        "--output",
        default="data/calibration_table.npz",
        help="Output .npz path (default: data/calibration_table.npz)",
    )
    parser.add_argument(
        "--n-bands",
        type=int,
        default=N_BANDS,
        help=f"Number of Y-bands (default: {N_BANDS})",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=_MIN_VOXELS,
        help=f"Minimum voxels per (band, biome) cell (default: {_MIN_VOXELS})",
    )
    args = parser.parse_args()

    logger.info("Computing calibration table ...")
    factors, band_centers, band_edges, num_chunks = compute_calibration_table(
        args.cache_dir,
        n_bands=args.n_bands,
        min_voxels=args.min_voxels,
    )

    # Save.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        calibration_factors=factors,
        band_centers=band_centers,
        band_edges=band_edges,
        num_chunks=np.array(num_chunks),
    )
    logger.info("Saved calibration table to %s (shape=%s, %d chunks)", out_path, factors.shape, num_chunks)

    # Print summary.
    mc_y_min = -64
    print(f"\nCalibration factors summary ({num_chunks} chunks, {args.n_bands} bands):")
    print(f"  {'Ore':>10s}  {'Avg Factor':>10s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}")
    for ore_idx, name in enumerate(_ORE_NAMES):
        f = factors[ore_idx]
        # Weight by biome prevalence (approximate — just average)
        avg = f.mean()
        print(f"  {name:>10s}  {avg:10.4f}  {f.min():8.4f}  {f.max():8.4f}")

    print(f"\n  Per-band centers (MC Y):")
    for i, c in enumerate(band_centers):
        mc_y = mc_y_min + c
        print(f"    Band {i:2d}: MC Y={mc_y:6.0f}  (range {mc_y_min + band_edges[i]:.0f} to {mc_y_min + band_edges[i + 1]:.0f})")


if __name__ == "__main__":
    main()
