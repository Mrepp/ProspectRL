"""Evaluate AnalyticalPrior accuracy against cached real Minecraft chunks.

Compares the geological prior's P(ore|y,biome) predictions against
empirical ore frequencies from .npz chunk data.  Reports per-ore
Y-profile correlations, MAE, total count ratios, calibration metrics,
and per-biome breakdowns.

Usage::

    python -m prospect_rl.tools.evaluate_prior \\
        --cache-dir data/chunk_cache/combined \\
        --plot
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from prospect_rl.config import (
    NUM_BIOME_TYPES,
    NUM_ORE_TYPES,
    ORE_TYPES,
    BiomeType,
    BlockType,
)
from multiagent.geological_prior import AnalyticalPrior

logger = logging.getLogger(__name__)

_ORE_NAMES = ["coal", "iron", "gold", "diamond", "redstone", "emerald", "lapis", "copper"]
_BIOME_NAMES = {int(b): b.name for b in BiomeType}
_ORE_BT_VALUES = [int(bt) for bt in ORE_TYPES]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EmpiricalStats:
    """Accumulated voxel counts from real chunks."""

    ore_count: np.ndarray  # (NUM_ORE_TYPES, world_height, NUM_BIOME_TYPES) int64
    total_voxels: np.ndarray  # (world_height, NUM_BIOME_TYPES) int64
    solid_voxels: np.ndarray  # (world_height, NUM_BIOME_TYPES) int64
    num_chunks: int
    world_height: int


@dataclass
class OreComparison:
    """Per-ore comparison metrics."""

    ore_name: str
    ore_index: int
    # Per-biome metrics: biome_id -> value
    pearson: dict[int, float] = field(default_factory=dict)
    spearman: dict[int, float] = field(default_factory=dict)
    mae: dict[int, float] = field(default_factory=dict)
    total_predicted: dict[int, float] = field(default_factory=dict)
    total_empirical: dict[int, float] = field(default_factory=dict)
    count_ratio: dict[int, float] = field(default_factory=dict)
    kl_div: dict[int, float] = field(default_factory=dict)
    # Biome-weighted overall
    overall_pearson: float = 0.0
    overall_spearman: float = 0.0
    overall_mae: float = 0.0
    overall_count_ratio: float = 0.0
    overall_kl_div: float = 0.0


# ---------------------------------------------------------------------------
# Step 1: Accumulate empirical stats from cached chunks
# ---------------------------------------------------------------------------


def accumulate_empirical_stats(cache_dir: str) -> EmpiricalStats:
    """Load all .npz chunks and count ores per (ore, y, biome)."""
    cache_path = Path(cache_dir)
    files = sorted(cache_path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {cache_dir}")

    # Auto-detect world height from first chunk
    first = np.load(files[0], allow_pickle=False)
    world_height = first["blocks"].shape[1]
    logger.info(
        "Found %d chunks (world_height=%d) in %s",
        len(files), world_height, cache_dir,
    )

    ore_count = np.zeros((NUM_ORE_TYPES, world_height, NUM_BIOME_TYPES), dtype=np.int64)
    total_voxels = np.zeros((world_height, NUM_BIOME_TYPES), dtype=np.int64)
    solid_voxels = np.zeros((world_height, NUM_BIOME_TYPES), dtype=np.int64)
    num_chunks = 0

    for filepath in files:
        data = np.load(filepath, allow_pickle=False)
        blocks = data["blocks"]  # (sx, sy, sz) int8
        biome_map = data["biome_map"]  # (sx, sz) int8

        if blocks.shape[1] != world_height:
            logger.warning(
                "Skipping %s: height %d != expected %d",
                filepath.name, blocks.shape[1], world_height,
            )
            continue

        num_chunks += 1

        for biome_id in np.unique(biome_map):
            biome_id_c = int(np.clip(biome_id, 0, NUM_BIOME_TYPES - 1))
            col_mask = biome_map == biome_id
            xs, zs = np.where(col_mask)
            n_cols = len(xs)
            if n_cols == 0:
                continue

            # Extract columns: (n_cols, world_height)
            col_blocks = blocks[xs, :, zs]

            # Total voxels per Y
            total_voxels[:, biome_id_c] += n_cols

            # Solid voxels per Y
            solid_voxels[:, biome_id_c] += np.count_nonzero(col_blocks, axis=0)

            # Ore counts per Y
            for ore_idx, ore_bt in enumerate(_ORE_BT_VALUES):
                ore_count[ore_idx, :, biome_id_c] += np.sum(
                    col_blocks == ore_bt, axis=0,
                )

        if num_chunks % 50 == 0:
            logger.info("  processed %d / %d chunks", num_chunks, len(files))

    logger.info("Finished: %d chunks, %s total voxels", num_chunks, f"{total_voxels.sum():,}")

    return EmpiricalStats(
        ore_count=ore_count,
        total_voxels=total_voxels,
        solid_voxels=solid_voxels,
        num_chunks=num_chunks,
        world_height=world_height,
    )


# ---------------------------------------------------------------------------
# Step 3: Compute comparison metrics
# ---------------------------------------------------------------------------


def _safe_correlation(
    x: np.ndarray, y: np.ndarray, method: str = "pearson",
) -> float:
    """Compute correlation, returning NaN if data is degenerate."""
    # Need at least 3 points and some variance in both arrays
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan")
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    if method == "pearson":
        r, _ = scipy_stats.pearsonr(x, y)
    else:
        r, _ = scipy_stats.spearmanr(x, y)
    return float(r)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) with epsilon smoothing. P = empirical, Q = prior."""
    eps = 1e-12
    # Normalize to distributions
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum < eps or q_sum < eps:
        return float("nan")
    p_norm = p / p_sum + eps
    q_norm = q / q_sum + eps
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    return float(np.sum(p_norm * np.log(p_norm / q_norm)))


def compute_comparison(
    prior: AnalyticalPrior,
    stats: EmpiricalStats,
) -> list[OreComparison]:
    """Compare prior predictions against empirical frequencies."""
    results: list[OreComparison] = []
    prior_table = prior._table  # (8, H, 5)
    h = stats.world_height

    # Empirical frequency: ore_count / total_voxels (per voxel)
    emp_freq = np.zeros_like(prior_table)
    for ore_idx in range(NUM_ORE_TYPES):
        for biome_id in range(NUM_BIOME_TYPES):
            denom = stats.total_voxels[:, biome_id].astype(np.float64)
            denom[denom == 0] = 1.0  # avoid div-by-zero; numerator is also 0
            emp_freq[ore_idx, :, biome_id] = (
                stats.ore_count[ore_idx, :, biome_id].astype(np.float64) / denom
            )

    for ore_idx, name in enumerate(_ORE_NAMES):
        comp = OreComparison(ore_name=name, ore_index=ore_idx)

        biome_weights: dict[int, float] = {}

        for biome_id in range(NUM_BIOME_TYPES):
            biome_voxels = float(stats.total_voxels[:, biome_id].sum())
            if biome_voxels < 1.0:
                continue  # no data for this biome

            prior_y = prior_table[ore_idx, :, biome_id]
            emp_y = emp_freq[ore_idx, :, biome_id]

            # Only compute correlation where we have actual data
            valid = stats.total_voxels[:, biome_id] > 0
            prior_valid = prior_y[valid]
            emp_valid = emp_y[valid]

            # Skip trivially zero pairs (e.g., emerald in PLAINS)
            if prior_valid.sum() < 1e-12 and emp_valid.sum() < 1e-12:
                continue

            biome_weights[biome_id] = biome_voxels

            comp.pearson[biome_id] = _safe_correlation(prior_valid, emp_valid, "pearson")
            comp.spearman[biome_id] = _safe_correlation(prior_valid, emp_valid, "spearman")
            comp.mae[biome_id] = float(np.mean(np.abs(prior_valid - emp_valid)))
            comp.total_predicted[biome_id] = float(prior_y.sum())
            comp.total_empirical[biome_id] = float(emp_y.sum())
            if comp.total_empirical[biome_id] > 1e-12:
                comp.count_ratio[biome_id] = (
                    comp.total_predicted[biome_id] / comp.total_empirical[biome_id]
                )
            else:
                comp.count_ratio[biome_id] = float("nan")
            comp.kl_div[biome_id] = _kl_divergence(emp_valid, prior_valid)

        # Biome-weighted overall metrics
        total_w = sum(biome_weights.values())
        if total_w > 0:
            for attr in ("pearson", "spearman", "mae", "kl_div"):
                per_biome = getattr(comp, attr)
                weighted = sum(
                    per_biome[b] * biome_weights[b]
                    for b in biome_weights
                    if b in per_biome and np.isfinite(per_biome[b])
                )
                denom = sum(
                    biome_weights[b]
                    for b in biome_weights
                    if b in per_biome and np.isfinite(per_biome[b])
                )
                setattr(comp, f"overall_{attr}", weighted / denom if denom > 0 else float("nan"))

            total_pred = sum(
                comp.total_predicted.get(b, 0) * biome_weights[b]
                for b in biome_weights
            )
            total_emp = sum(
                comp.total_empirical.get(b, 0) * biome_weights[b]
                for b in biome_weights
            )
            comp.overall_count_ratio = total_pred / total_emp if total_emp > 1e-12 else float("nan")

        results.append(comp)

    return results


# ---------------------------------------------------------------------------
# Step 4: Calibration
# ---------------------------------------------------------------------------


def compute_calibration(
    prior: AnalyticalPrior,
    stats: EmpiricalStats,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin prior predictions and compute observed frequencies.

    Returns (predicted_mean, observed_freq, bin_voxel_counts) each of shape (n_bins,).
    """
    prior_table = prior._table
    h = stats.world_height

    # Flatten all (ore, y, biome) entries
    pred_flat = prior_table.ravel().astype(np.float64)

    emp_flat = np.zeros_like(pred_flat)
    weight_flat = np.zeros_like(pred_flat)

    idx = 0
    for ore_idx in range(NUM_ORE_TYPES):
        for y in range(h):
            for biome_id in range(NUM_BIOME_TYPES):
                denom = float(stats.total_voxels[y, biome_id])
                if denom > 0:
                    emp_flat[idx] = stats.ore_count[ore_idx, y, biome_id] / denom
                    weight_flat[idx] = denom
                idx += 1

    # Create bins from 0 to max prediction
    max_pred = pred_flat.max() + 1e-12
    bin_edges = np.linspace(0, max_pred, n_bins + 1)

    predicted_mean = np.zeros(n_bins)
    observed_freq = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (pred_flat >= lo) & (pred_flat <= hi)
        else:
            mask = (pred_flat >= lo) & (pred_flat < hi)

        w = weight_flat[mask]
        if w.sum() > 0:
            predicted_mean[i] = np.average(pred_flat[mask], weights=w)
            observed_freq[i] = np.average(emp_flat[mask], weights=w)
            bin_counts[i] = w.sum()

    return predicted_mean, observed_freq, bin_counts


# ---------------------------------------------------------------------------
# Step 5: Console output
# ---------------------------------------------------------------------------


def print_comparison(
    results: list[OreComparison],
    stats: EmpiricalStats,
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray],
    prior: AnalyticalPrior,
) -> None:
    """Pretty-print evaluation results."""
    total_vox = stats.total_voxels.sum()
    print(f"\n{'=' * 68}")
    print(f"  Prior Accuracy Evaluation ({stats.num_chunks} chunks, "
          f"{total_vox:,} voxels)")
    print(f"{'=' * 68}")

    # Biome data coverage
    print("\n  Biome coverage:")
    for b in range(NUM_BIOME_TYPES):
        bv = stats.total_voxels[:, b].sum()
        name = _BIOME_NAMES.get(b, f"biome_{b}")
        if bv > 0:
            print(f"    {name:>20s}: {bv:>14,} voxels ({bv / total_vox * 100:5.1f}%)")
        else:
            print(f"    {name:>20s}: {'(no data)':>14s}")

    # Per-ore overall table
    print(f"\n  Per-Ore Y-Profile Accuracy (biome-weighted):")
    print(f"    {'Ore':>10s}  {'Pearson':>8s}  {'Spearman':>8s}  "
          f"{'MAE':>10s}  {'Pred/Act':>8s}  {'KL-div':>8s}")
    print(f"    {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 10}  {'-' * 8}  {'-' * 8}")
    for comp in results:
        if not comp.pearson:
            print(f"    {comp.ore_name:>10s}  {'(no data)':>8s}")
            continue
        p = comp.overall_pearson
        s = comp.overall_spearman
        m = comp.overall_mae
        r = comp.overall_count_ratio
        k = comp.overall_kl_div
        print(
            f"    {comp.ore_name:>10s}  "
            f"{_fmt(p, 8)}  {_fmt(s, 8)}  "
            f"{_fmt(m, 10, 6)}  {_fmt(r, 8, 3)}  {_fmt(k, 8, 4)}"
        )

    # Per-biome breakdown
    print(f"\n  Per-Ore Per-Biome Breakdown:")
    for comp in results:
        if not comp.pearson:
            continue
        for biome_id in sorted(comp.pearson.keys()):
            bname = _BIOME_NAMES.get(biome_id, f"biome_{biome_id}")
            p = comp.pearson[biome_id]
            m = comp.mae[biome_id]
            r = comp.count_ratio.get(biome_id, float("nan"))
            tp = comp.total_predicted.get(biome_id, 0)
            te = comp.total_empirical.get(biome_id, 0)
            print(
                f"    {comp.ore_name:>10s} / {bname:<18s}  "
                f"Pearson={_fmt(p, 6)}  MAE={_fmt(m, 8, 6)}  "
                f"Pred/Act={_fmt(r, 6, 3)}  "
                f"(pred={tp:.3f}  act={te:.3f} ores/col)"
            )

    # Calibration
    pred_mean, obs_freq, bin_counts = calibration
    print(f"\n  Calibration ({len(pred_mean)} bins):")
    print(f"    {'Predicted':>12s}  {'Observed':>12s}  {'N-voxels':>14s}  {'Ratio':>8s}")
    print(f"    {'-' * 12}  {'-' * 12}  {'-' * 14}  {'-' * 8}")
    for i in range(len(pred_mean)):
        if bin_counts[i] > 0:
            ratio = obs_freq[i] / pred_mean[i] if pred_mean[i] > 1e-15 else float("nan")
            print(
                f"    {pred_mean[i]:12.7f}  {obs_freq[i]:12.7f}  "
                f"{bin_counts[i]:14,.0f}  {_fmt(ratio, 8, 3)}"
            )

    # Worst Y-levels
    prior_table = prior._table
    print(f"\n  Top-10 Worst Y-levels (largest |prior - actual|):")
    errors: list[tuple[float, int, int, int, float, float]] = []
    for ore_idx in range(NUM_ORE_TYPES):
        for biome_id in range(NUM_BIOME_TYPES):
            for y in range(stats.world_height):
                denom = float(stats.total_voxels[y, biome_id])
                if denom < 100:  # need meaningful sample
                    continue
                emp = stats.ore_count[ore_idx, y, biome_id] / denom
                pred = float(prior_table[ore_idx, y, biome_id])
                err = abs(pred - emp)
                errors.append((err, ore_idx, y, biome_id, pred, emp))

    errors.sort(reverse=True)
    for err, ore_idx, y, biome_id, pred, emp in errors[:10]:
        bname = _BIOME_NAMES.get(biome_id, f"biome_{biome_id}")
        delta = pred - emp
        sign = "+" if delta >= 0 else ""
        print(
            f"    {_ORE_NAMES[ore_idx]:>10s}  y={y:3d}  {bname:<18s}  "
            f"prior={pred:.6f}  actual={emp:.6f}  "
            f"(delta={sign}{delta:.6f})"
        )

    print()


def _print_ab_comparison(
    results_new: list[OreComparison],
    results_old: list[OreComparison],
) -> None:
    """Print side-by-side comparison of corrected vs uncorrected prior."""
    print(f"\n{'=' * 68}")
    print("  Solid Fraction Correction: Before vs After")
    print(f"{'=' * 68}")
    print(f"    {'Ore':>10s}  {'Old Pred/Act':>12s}  {'New Pred/Act':>12s}  {'Improvement':>12s}")
    print(f"    {'-' * 10}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
    for old, new in zip(results_old, results_new):
        r_old = old.overall_count_ratio
        r_new = new.overall_count_ratio
        if np.isfinite(r_old) and np.isfinite(r_new) and r_old > 0:
            # Improvement: how much closer to 1.0
            old_err = abs(r_old - 1.0)
            new_err = abs(r_new - 1.0)
            if old_err > 1e-6:
                pct = (old_err - new_err) / old_err * 100
                imp_str = f"{pct:+.0f}%"
            else:
                imp_str = "N/A"
        else:
            imp_str = "N/A"
        print(
            f"    {old.ore_name:>10s}  "
            f"{_fmt(r_old, 12, 3)}  "
            f"{_fmt(r_new, 12, 3)}  "
            f"{imp_str:>12s}"
        )
    print()


def _print_calibration_comparison(
    results_cal: list[OreComparison],
    results_uncal: list[OreComparison],
) -> None:
    """Print side-by-side comparison of calibrated vs uncalibrated."""
    print(f"\n{'=' * 68}")
    print("  Empirical Calibration: Before vs After")
    print(f"{'=' * 68}")
    hdr = (
        f"    {'Ore':>10s}  {'Uncalibrated':>12s}  "
        f"{'Calibrated':>12s}  {'Improvement':>12s}"
    )
    print(hdr)
    print(
        f"    {'-' * 10}  {'-' * 12}  "
        f"{'-' * 12}  {'-' * 12}"
    )
    for uncal, cal in zip(results_uncal, results_cal):
        r_uncal = uncal.overall_count_ratio
        r_cal = cal.overall_count_ratio
        if (
            np.isfinite(r_uncal)
            and np.isfinite(r_cal)
            and r_uncal > 0
        ):
            old_err = abs(r_uncal - 1.0)
            new_err = abs(r_cal - 1.0)
            if old_err > 1e-6:
                pct = (old_err - new_err) / old_err * 100
                imp_str = f"{pct:+.0f}%"
            else:
                imp_str = "N/A"
        else:
            imp_str = "N/A"
        print(
            f"    {uncal.ore_name:>10s}  "
            f"{_fmt(r_uncal, 12, 3)}  "
            f"{_fmt(r_cal, 12, 3)}  "
            f"{imp_str:>12s}"
        )
    print()


def _fmt(val: float, width: int, decimals: int = 4) -> str:
    """Format float or NaN to fixed width."""
    if not np.isfinite(val):
        return f"{'N/A':>{width}s}"
    return f"{val:>{width}.{decimals}f}"


# ---------------------------------------------------------------------------
# Step 6: Plots
# ---------------------------------------------------------------------------


def plot_comparison(
    results: list[OreComparison],
    prior: AnalyticalPrior,
    stats: EmpiricalStats,
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_dir: str,
) -> None:
    """Generate and save matplotlib comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prior_table = prior._table
    h = stats.world_height

    # --- Figure 1: Y-profile comparison (2x4 grid) ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Prior vs Empirical Ore Y-Profiles", fontsize=14)

    biome_colors = {0: "tab:blue", 1: "tab:green", 2: "tab:orange", 3: "tab:red", 4: "tab:purple"}
    y_levels = np.arange(h)

    for ore_idx, ax in enumerate(axes.flat):
        name = _ORE_NAMES[ore_idx]
        ax.set_title(name.capitalize(), fontsize=11)

        has_data = False
        for biome_id in range(NUM_BIOME_TYPES):
            bv = stats.total_voxels[:, biome_id]
            if bv.sum() < 1:
                continue
            prior_y = prior_table[ore_idx, :, biome_id]
            denom = bv.astype(np.float64)
            denom[denom == 0] = 1.0
            emp_y = stats.ore_count[ore_idx, :, biome_id].astype(np.float64) / denom

            if prior_y.sum() < 1e-12 and emp_y.sum() < 1e-12:
                continue

            has_data = True
            bname = _BIOME_NAMES.get(biome_id, "?")
            color = biome_colors.get(biome_id, "gray")
            ax.plot(y_levels, prior_y, color=color, alpha=0.8, linewidth=1.2,
                    label=f"{bname} prior")
            ax.plot(y_levels, emp_y, color=color, alpha=0.5, linewidth=1.2,
                    linestyle="--", label=f"{bname} actual")

        if has_data:
            ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("Y-level (sim)")
        ax.set_ylabel("P(ore)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    plt.tight_layout()
    path1 = out / "y_profiles.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path1)

    # --- Figure 2: Calibration plot ---
    pred_mean, obs_freq, bin_counts = calibration
    mask = bin_counts > 0
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Prior Calibration Plot", fontsize=13)
    max_val = max(pred_mean[mask].max(), obs_freq[mask].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="Perfect")
    sizes = np.clip(bin_counts[mask] / bin_counts[mask].max() * 200, 10, 200)
    ax.scatter(pred_mean[mask], obs_freq[mask], s=sizes, alpha=0.7, c="steelblue")
    ax.set_xlabel("Predicted P(ore)")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    plt.tight_layout()
    path2 = out / "calibration.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path2)

    # --- Figure 3: Error heatmap ---
    # Biome-weighted average error across all biomes
    err_map = np.zeros((NUM_ORE_TYPES, h), dtype=np.float64)
    for ore_idx in range(NUM_ORE_TYPES):
        weighted_err = np.zeros(h, dtype=np.float64)
        total_w = np.zeros(h, dtype=np.float64)
        for biome_id in range(NUM_BIOME_TYPES):
            bv = stats.total_voxels[:, biome_id].astype(np.float64)
            denom = bv.copy()
            denom[denom == 0] = 1.0
            emp_y = stats.ore_count[ore_idx, :, biome_id].astype(np.float64) / denom
            prior_y = prior_table[ore_idx, :, biome_id]
            weighted_err += (prior_y - emp_y) * bv
            total_w += bv
        total_w[total_w == 0] = 1.0
        err_map[ore_idx, :] = weighted_err / total_w

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_title("Prior Error Heatmap (prior - actual, biome-weighted)", fontsize=13)
    vmax = np.abs(err_map).max()
    im = ax.imshow(
        err_map, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[0, h, NUM_ORE_TYPES - 0.5, -0.5],
    )
    ax.set_yticks(range(NUM_ORE_TYPES))
    ax.set_yticklabels([n.capitalize() for n in _ORE_NAMES])
    ax.set_xlabel("Y-level (sim)")
    fig.colorbar(im, ax=ax, label="Prior - Actual")
    plt.tight_layout()
    path3 = out / "error_heatmap.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path3)

    # --- Figure 4: Total count bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Total Ores Per Column: Predicted vs Actual", fontsize=13)
    x = np.arange(NUM_ORE_TYPES)
    width = 0.35

    pred_totals = []
    emp_totals = []
    for ore_idx in range(NUM_ORE_TYPES):
        # Weighted average across biomes
        pred_sum = 0.0
        emp_sum = 0.0
        total_w = 0.0
        for biome_id in range(NUM_BIOME_TYPES):
            bv = float(stats.total_voxels[:, biome_id].sum())
            if bv < 1:
                continue
            n_cols = bv / stats.world_height  # approximate column count
            pred_sum += prior_table[ore_idx, :, biome_id].sum() * n_cols
            ore_total = float(stats.ore_count[ore_idx, :, biome_id].sum())
            emp_sum += ore_total
            total_w += n_cols
        # Average ores per column
        pred_totals.append(pred_sum / total_w if total_w > 0 else 0)
        emp_totals.append(emp_sum / total_w if total_w > 0 else 0)

    ax.bar(x - width / 2, pred_totals, width, label="Predicted", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, emp_totals, width, label="Actual", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in _ORE_NAMES], rotation=30)
    ax.set_ylabel("Ores per column")
    ax.legend()
    plt.tight_layout()
    path4 = out / "total_counts.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path4)

    # --- Figure 5: Solid fraction profile ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Solid Fraction by Y-Level and Biome", fontsize=13)
    for biome_id in range(NUM_BIOME_TYPES):
        bv = stats.total_voxels[:, biome_id]
        if bv.sum() < 1:
            continue
        denom = bv.astype(np.float64)
        denom[denom == 0] = 1.0
        solid_frac = stats.solid_voxels[:, biome_id].astype(np.float64) / denom
        bname = _BIOME_NAMES.get(biome_id, "?")
        color = biome_colors.get(biome_id, "gray")
        ax.plot(y_levels, solid_frac, label=bname, color=color, alpha=0.8)
    ax.set_xlabel("Y-level (sim)")
    ax.set_ylabel("Solid fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    plt.tight_layout()
    path5 = out / "solid_fraction.png"
    fig.savefig(path5, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path5)

    print(f"  Plots saved to {out}/")


# ---------------------------------------------------------------------------
# Step 7: CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate AnalyticalPrior accuracy against real MC chunks.",
    )
    parser.add_argument(
        "--cache-dir", default="data/chunk_cache/combined",
        help="Directory containing .npz cache files",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib comparison plots",
    )
    parser.add_argument(
        "--plot-dir", default="data/prior_evaluation",
        help="Directory to save plots (default: data/prior_evaluation/)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare corrected prior vs uncorrected (no solid fraction)",
    )
    parser.add_argument(
        "--calibrated", action="store_true",
        help="Compare calibrated prior vs uncalibrated",
    )
    args = parser.parse_args()

    # Step 1: Accumulate empirical stats
    logger.info("Loading cached chunks from %s ...", args.cache_dir)
    stats = accumulate_empirical_stats(args.cache_dir)

    # Step 2: Instantiate prior(s)
    logger.info("Building AnalyticalPrior (world_height=%d) ...", stats.world_height)
    prior = AnalyticalPrior(world_height=stats.world_height)

    # Step 3: Compare
    logger.info("Computing comparison metrics ...")
    results = compute_comparison(prior, stats)

    # Step 4: Calibration
    calibration = compute_calibration(prior, stats)

    # Step 5: Print
    print_comparison(results, stats, calibration, prior)

    # Step 5b: A/B comparison (solid fraction)
    if args.compare:
        logger.info("Building uncorrected prior for comparison ...")
        prior_old = AnalyticalPrior(
            world_height=stats.world_height,
            apply_solid_correction=False,
        )
        results_old = compute_comparison(prior_old, stats)
        _print_ab_comparison(results, results_old)

    # Step 5c: A/B comparison (empirical calibration)
    if args.calibrated:
        logger.info(
            "Building uncalibrated prior for comparison ...",
        )
        prior_uncal = AnalyticalPrior(
            world_height=stats.world_height,
            apply_calibration=False,
        )
        results_uncal = compute_comparison(prior_uncal, stats)
        _print_calibration_comparison(results, results_uncal)

    # Step 6: Plots
    if args.plot:
        plot_comparison(results, prior, stats, calibration, args.plot_dir)


if __name__ == "__main__":
    main()
