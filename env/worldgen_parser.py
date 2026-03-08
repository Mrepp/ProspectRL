"""Parse vanilla Minecraft 1.21.11 worldgen JSON into ore placement data.

Reads ``data/worldgen/`` configured features, placed features, and biome
definitions to produce structured ore placement parameters.  This module
has **no** imports from ``prospect_rl.config`` to avoid circular dependencies.

The parsed :class:`WorldgenData` is consumed by:
- ``config.py`` to generate ``ORE_SPAWN_CONFIGS``
- ``multiagent/geological_prior.py`` to build the analytical prior table
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeightDistribution:
    """Parsed ``minecraft:height_range`` from a placed-feature JSON."""

    dist_type: str  # "uniform" or "trapezoid"
    min_inclusive: int  # Resolved absolute MC Y
    max_inclusive: int  # Resolved absolute MC Y
    plateau: int = 0  # Trapezoid plateau width (0 = triangle)


@dataclass(frozen=True)
class AttemptModel:
    """How many placement attempts per chunk for a placed feature.

    Supports three Minecraft attempt types:
    - ``count_fixed``: constant integer count
    - ``count_uniform``: uniform random integer in [min_val, max_val]
    - ``rarity_filter``: 1/chance probability per chunk
    """

    model_type: str  # "count_fixed", "count_uniform", "rarity_filter"
    value: float  # Fixed count, OR rarity chance
    min_val: int = 0  # For count_uniform
    max_val: int = 0  # For count_uniform

    @property
    def expected_attempts(self) -> float:
        if self.model_type == "count_fixed":
            return self.value
        elif self.model_type == "count_uniform":
            return (self.min_val + self.max_val) / 2.0
        elif self.model_type == "rarity_filter":
            return 1.0 / self.value if self.value > 0 else 0.0
        return self.value


@dataclass(frozen=True)
class ConfiguredOreFeature:
    """Parsed ``configured_feature/ore_*.json``."""

    name: str  # e.g. "ore_diamond_small"
    size: int  # vein size in blocks
    discard_chance_on_air_exposure: float  # 0.0 to 1.0
    ore_block: str  # e.g. "coal_ore" (no "minecraft:" prefix)


@dataclass
class PlacedOreFeature:
    """Parsed ``placed_feature/ore_*.json``, linked to its configured feature."""

    name: str  # e.g. "ore_diamond"
    configured_feature_name: str  # e.g. "ore_diamond_small"
    attempt_model: AttemptModel
    height_distribution: HeightDistribution
    configured: ConfiguredOreFeature | None = None  # Set by link_features()


@dataclass
class BiomeOreProfile:
    """Which placed ore features are active in a biome (from features[6])."""

    biome_name: str
    ore_features: list[str] = field(default_factory=list)


@dataclass
class WorldgenData:
    """Bundle of all parsed worldgen data."""

    configured_features: dict[str, ConfiguredOreFeature]
    placed_features: dict[str, PlacedOreFeature]
    biome_profiles: dict[str, BiomeOreProfile]
    min_y: int  # -64 for Overworld
    max_y: int  # 320 for Overworld


# ---------------------------------------------------------------------------
# Ore blocks we care about (maps MC block name → our ore name)
# ---------------------------------------------------------------------------

# Maps MC block names to a canonical ore name (used to filter non-ore features).
# Both stone and deepslate variants map to the same canonical name.
TRACKED_ORE_BLOCKS: dict[str, str] = {
    "coal_ore": "coal",
    "deepslate_coal_ore": "coal",
    "iron_ore": "iron",
    "deepslate_iron_ore": "iron",
    "gold_ore": "gold",
    "deepslate_gold_ore": "gold",
    "diamond_ore": "diamond",
    "deepslate_diamond_ore": "diamond",
    "redstone_ore": "redstone",
    "deepslate_redstone_ore": "redstone",
    "emerald_ore": "emerald",
    "deepslate_emerald_ore": "emerald",
    "lapis_ore": "lapis",
    "deepslate_lapis_ore": "lapis",
    "copper_ore": "copper",
    "deepslate_copper_ore": "copper",
}

# Nether / End biomes — skip when parsing overworld ores.
_NON_OVERWORLD_BIOMES = {
    "nether_wastes", "crimson_forest", "warped_forest",
    "soul_sand_valley", "basalt_deltas",
    "the_end", "end_barrens", "end_highlands",
    "end_midlands", "small_end_islands", "the_void",
}


# ---------------------------------------------------------------------------
# Vertical anchor resolution
# ---------------------------------------------------------------------------


def resolve_vertical_anchor(
    anchor: dict, min_y: int = -64, max_y: int = 320,
) -> int:
    """Resolve a Minecraft vertical anchor to an absolute Y-coordinate.

    Anchor types:
    - ``{"absolute": N}`` → N
    - ``{"above_bottom": N}`` → min_y + N
    - ``{"below_top": N}`` → max_y - N
    """
    if "absolute" in anchor:
        return int(anchor["absolute"])
    if "above_bottom" in anchor:
        return min_y + int(anchor["above_bottom"])
    if "below_top" in anchor:
        return max_y - int(anchor["below_top"])
    raise ValueError(f"Unknown vertical anchor type: {anchor}")


# ---------------------------------------------------------------------------
# Height distribution PMF
# ---------------------------------------------------------------------------


def compute_height_pmf(
    dist: HeightDistribution,
    world_min_y: int = -64,
    world_max_y: int = 320,
) -> dict[int, float]:
    """Compute a normalized PMF over integer MC Y-levels.

    Returns ``{mc_y: probability}`` where probabilities sum to 1.0.
    The distribution range is clamped to ``[world_min_y, world_max_y]``.
    """
    lo = max(dist.min_inclusive, world_min_y)
    hi = min(dist.max_inclusive, world_max_y)

    if lo > hi:
        return {}

    if dist.dist_type == "uniform":
        n = hi - lo + 1
        p = 1.0 / n
        return {y: p for y in range(lo, hi + 1)}

    # Trapezoid (plateau=0 → triangle, symmetric about midpoint)
    mid = (lo + hi) / 2.0
    half_range = (hi - lo) / 2.0
    half_plateau = dist.plateau / 2.0

    if half_range <= 0:
        return {lo: 1.0}

    ramp = half_range - half_plateau
    if ramp <= 0:
        # Entire range is plateau
        n = hi - lo + 1
        return {y: 1.0 / n for y in range(lo, hi + 1)}

    weights: dict[int, float] = {}
    total = 0.0
    for y in range(lo, hi + 1):
        d = abs(y - mid)
        if d <= half_plateau:
            w = 1.0
        else:
            w = max(0.0, 1.0 - (d - half_plateau) / ramp)
        weights[y] = w
        total += w

    if total > 0:
        for y in weights:
            weights[y] /= total

    return weights


# ---------------------------------------------------------------------------
# JSON parsing functions
# ---------------------------------------------------------------------------


def parse_configured_features(
    data_dir: Path,
) -> dict[str, ConfiguredOreFeature]:
    """Parse all ``configured_feature/ore_*.json`` files.

    Returns a dict keyed by feature name (filename stem).
    Only includes features whose ``type`` is ``minecraft:ore`` and whose
    first target block is a tracked ore.
    """
    cf_dir = data_dir / "configured_feature"
    result: dict[str, ConfiguredOreFeature] = {}

    for path in sorted(cf_dir.glob("ore_*.json")):
        with open(path) as f:
            data = json.load(f)

        if data.get("type") != "minecraft:ore":
            continue

        config = data.get("config", {})
        size = config.get("size", 0)
        discard = config.get("discard_chance_on_air_exposure", 0.0)

        targets = config.get("targets", [])
        if not targets:
            continue

        # Extract ore block name from first target
        state = targets[0].get("state", {})
        block_name = state.get("Name", "")
        # Strip "minecraft:" prefix
        if block_name.startswith("minecraft:"):
            block_name = block_name[len("minecraft:"):]

        # Only track our 8 ore types
        if block_name not in TRACKED_ORE_BLOCKS:
            continue

        name = path.stem  # e.g. "ore_diamond_small"
        result[name] = ConfiguredOreFeature(
            name=name,
            size=size,
            discard_chance_on_air_exposure=discard,
            ore_block=block_name,
        )

    return result


def _parse_attempt_model(placement: list[dict]) -> AttemptModel:
    """Extract the attempt model from a placed feature's placement array."""
    for modifier in placement:
        mod_type = modifier.get("type", "")

        if mod_type == "minecraft:count":
            count = modifier.get("count", 0)
            if isinstance(count, dict):
                # Uniform IntProvider, e.g. {"type": "minecraft:uniform",
                #   "min_inclusive": 0, "max_inclusive": 1}
                min_val = int(count.get("min_inclusive", 0))
                max_val = int(count.get("max_inclusive", 0))
                return AttemptModel(
                    model_type="count_uniform",
                    value=0.0,
                    min_val=min_val,
                    max_val=max_val,
                )
            else:
                return AttemptModel(
                    model_type="count_fixed",
                    value=float(count),
                )

        if mod_type == "minecraft:rarity_filter":
            chance = modifier.get("chance", 1)
            return AttemptModel(
                model_type="rarity_filter",
                value=float(chance),
            )

    # Fallback: single attempt
    return AttemptModel(model_type="count_fixed", value=1.0)


def _parse_height_range(
    placement: list[dict],
    min_y: int,
    max_y: int,
) -> HeightDistribution:
    """Extract the height distribution from a placement array."""
    for modifier in placement:
        if modifier.get("type") != "minecraft:height_range":
            continue

        height = modifier.get("height", {})
        dist_type = height.get("type", "minecraft:uniform")
        # Strip "minecraft:" prefix
        if dist_type.startswith("minecraft:"):
            dist_type = dist_type[len("minecraft:"):]

        lo = resolve_vertical_anchor(
            height.get("min_inclusive", {"absolute": min_y}),
            min_y=min_y, max_y=max_y,
        )
        hi = resolve_vertical_anchor(
            height.get("max_inclusive", {"absolute": max_y}),
            min_y=min_y, max_y=max_y,
        )
        plateau = height.get("plateau", 0)

        return HeightDistribution(
            dist_type=dist_type,
            min_inclusive=lo,
            max_inclusive=hi,
            plateau=plateau,
        )

    # Fallback: full world range, uniform
    return HeightDistribution(
        dist_type="uniform",
        min_inclusive=min_y,
        max_inclusive=max_y,
    )


def parse_placed_features(
    data_dir: Path,
    min_y: int = -64,
    max_y: int = 320,
) -> dict[str, PlacedOreFeature]:
    """Parse all ``placed_feature/ore_*.json`` files.

    Returns a dict keyed by placed feature name (filename stem).
    """
    pf_dir = data_dir / "placed_feature"
    result: dict[str, PlacedOreFeature] = {}

    for path in sorted(pf_dir.glob("ore_*.json")):
        with open(path) as f:
            data = json.load(f)

        # The "feature" field references the configured feature
        feature_ref = data.get("feature", "")
        if feature_ref.startswith("minecraft:"):
            feature_ref = feature_ref[len("minecraft:"):]

        placement = data.get("placement", [])
        attempt = _parse_attempt_model(placement)
        height = _parse_height_range(placement, min_y, max_y)

        name = path.stem
        result[name] = PlacedOreFeature(
            name=name,
            configured_feature_name=feature_ref,
            attempt_model=attempt,
            height_distribution=height,
        )

    return result


def parse_biome_ore_profiles(
    data_dir: Path,
) -> dict[str, BiomeOreProfile]:
    """Parse biome JSONs and extract their ore feature lists (features[6]).

    Skips nether/end biomes and metadata files (_list.json, _all.json).
    Returns a dict keyed by biome name (filename stem).
    """
    biome_dir = data_dir / "biome"
    result: dict[str, BiomeOreProfile] = {}

    for path in sorted(biome_dir.glob("*.json")):
        name = path.stem
        if name.startswith("_"):
            continue
        if name in _NON_OVERWORLD_BIOMES:
            continue

        with open(path) as f:
            data = json.load(f)

        features = data.get("features", [])
        # Slot 6 is the underground ore generation step
        if len(features) <= 6:
            continue

        ore_step = features[6]
        ore_features: list[str] = []
        for feat in ore_step:
            feat_name = feat
            if feat_name.startswith("minecraft:"):
                feat_name = feat_name[len("minecraft:"):]
            # Only keep ore_* features (skip disk_*, underwater_*, etc.)
            if feat_name.startswith("ore_"):
                ore_features.append(feat_name)

        result[name] = BiomeOreProfile(
            biome_name=name,
            ore_features=ore_features,
        )

    return result


def link_features(
    configured: dict[str, ConfiguredOreFeature],
    placed: dict[str, PlacedOreFeature],
) -> None:
    """Link each placed feature to its configured feature (in-place)."""
    for pf in placed.values():
        cf = configured.get(pf.configured_feature_name)
        if cf is not None:
            # PlacedOreFeature is not frozen, so we can assign directly
            pf.configured = cf


# ---------------------------------------------------------------------------
# Biome classification → 5-biome model
# ---------------------------------------------------------------------------

# BiomeType enum values (mirrors config.py without importing it)
_PLAINS = 0
_MOUNTAINS = 1
_BADLANDS = 2
_DRIPSTONE_CAVES = 3
_LUSH_CAVES = 4


def classify_biome(profile: BiomeOreProfile) -> int:
    """Map a vanilla biome to the 5-biome model ID.

    Classification is based on which ore features differ from the standard
    set (plains-like biomes):
    - MOUNTAINS: has ``ore_emerald``
    - BADLANDS: has ``ore_gold_extra``
    - DRIPSTONE_CAVES: has ``ore_copper_large`` (replaces ``ore_copper``)
    - LUSH_CAVES: has ``ore_clay`` (same 8-ore set as PLAINS)
    - PLAINS: everything else
    """
    features = set(profile.ore_features)
    if "ore_emerald" in features:
        return _MOUNTAINS
    if "ore_gold_extra" in features:
        return _BADLANDS
    if "ore_copper_large" in features:
        return _DRIPSTONE_CAVES
    if profile.biome_name == "lush_caves":
        return _LUSH_CAVES
    return _PLAINS


def get_ore_features_per_biome_group(
    biome_profiles: dict[str, BiomeOreProfile],
) -> dict[int, set[str]]:
    """Return set of placed feature names active per 5-biome group.

    For each biome group, unions all placed ore feature names from all
    vanilla biomes that classify into that group.
    """
    groups: dict[int, set[str]] = {i: set() for i in range(5)}

    for profile in biome_profiles.values():
        group_id = classify_biome(profile)
        groups[group_id].update(profile.ore_features)

    return groups


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_CACHED_WORLDGEN: WorldgenData | None = None


def load_worldgen(
    data_dir: str | Path = "data/worldgen",
) -> WorldgenData:
    """Parse all worldgen JSON and return a :class:`WorldgenData` bundle.

    Results are cached at module level — subsequent calls return the
    same object.
    """
    global _CACHED_WORLDGEN
    if _CACHED_WORLDGEN is not None:
        return _CACHED_WORLDGEN

    data_dir = Path(data_dir)

    # Read Overworld dimension parameters
    noise_path = data_dir / "noise_settings" / "overworld.json"
    with open(noise_path) as f:
        noise_data = json.load(f)
    noise = noise_data.get("noise", {})
    min_y = noise.get("min_y", -64)
    height = noise.get("height", 384)
    max_y = min_y + height  # 320

    # Parse the three JSON families
    configured = parse_configured_features(data_dir)
    placed = parse_placed_features(data_dir, min_y=min_y, max_y=max_y)
    biomes = parse_biome_ore_profiles(data_dir)

    # Link placed → configured
    link_features(configured, placed)

    wg = WorldgenData(
        configured_features=configured,
        placed_features=placed,
        biome_profiles=biomes,
        min_y=min_y,
        max_y=max_y,
    )

    _CACHED_WORLDGEN = wg
    return wg


def clear_cache() -> None:
    """Clear the cached worldgen data (useful for testing)."""
    global _CACHED_WORLDGEN
    _CACHED_WORLDGEN = None
