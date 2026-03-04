"""Mapping from Minecraft Java Edition block/biome names to ProspectRL types.

Used by the chunk cache tool (``tools/cache_chunks.py``) to convert
Amulet Core block data to our ``BlockType`` / ``BiomeType`` enums.

The mapping covers:
- All 8 ore types (normal + deepslate variants)
- Stone-family, dirt-family, and filler blocks
- Air variants (cave_air, void_air)
- Biome name → simplified BiomeType
"""

from __future__ import annotations

from prospect_rl.config import BiomeType, BlockType

# ---------------------------------------------------------------------------
# Minecraft block name → BlockType
# ---------------------------------------------------------------------------

MC_TO_BLOCKTYPE: dict[str, int] = {
    # Air
    "minecraft:air": BlockType.AIR,
    "minecraft:cave_air": BlockType.AIR,
    "minecraft:void_air": BlockType.AIR,
    # Bedrock
    "minecraft:bedrock": BlockType.BEDROCK,
    # Stone family
    "minecraft:stone": BlockType.STONE,
    "minecraft:cobblestone": BlockType.STONE,
    "minecraft:smooth_stone": BlockType.STONE,
    "minecraft:mossy_cobblestone": BlockType.STONE,
    "minecraft:infested_stone": BlockType.STONE,
    "minecraft:stone_bricks": BlockType.STONE,
    "minecraft:cracked_stone_bricks": BlockType.STONE,
    "minecraft:mossy_stone_bricks": BlockType.STONE,
    "minecraft:chiseled_stone_bricks": BlockType.STONE,
    # Deepslate family
    "minecraft:deepslate": BlockType.DEEPSLATE,
    "minecraft:cobbled_deepslate": BlockType.DEEPSLATE,
    "minecraft:polished_deepslate": BlockType.DEEPSLATE,
    "minecraft:deepslate_bricks": BlockType.DEEPSLATE,
    "minecraft:cracked_deepslate_bricks": BlockType.DEEPSLATE,
    "minecraft:deepslate_tiles": BlockType.DEEPSLATE,
    "minecraft:cracked_deepslate_tiles": BlockType.DEEPSLATE,
    "minecraft:chiseled_deepslate": BlockType.DEEPSLATE,
    "minecraft:infested_deepslate": BlockType.DEEPSLATE,
    # Ores — normal variants
    "minecraft:coal_ore": BlockType.COAL_ORE,
    "minecraft:iron_ore": BlockType.IRON_ORE,
    "minecraft:gold_ore": BlockType.GOLD_ORE,
    "minecraft:diamond_ore": BlockType.DIAMOND_ORE,
    "minecraft:redstone_ore": BlockType.REDSTONE_ORE,
    "minecraft:emerald_ore": BlockType.EMERALD_ORE,
    "minecraft:lapis_ore": BlockType.LAPIS_ORE,
    "minecraft:copper_ore": BlockType.COPPER_ORE,
    # Ores — deepslate variants
    "minecraft:deepslate_coal_ore": BlockType.COAL_ORE,
    "minecraft:deepslate_iron_ore": BlockType.IRON_ORE,
    "minecraft:deepslate_gold_ore": BlockType.GOLD_ORE,
    "minecraft:deepslate_diamond_ore": BlockType.DIAMOND_ORE,
    "minecraft:deepslate_redstone_ore": BlockType.REDSTONE_ORE,
    "minecraft:deepslate_emerald_ore": BlockType.EMERALD_ORE,
    "minecraft:deepslate_lapis_ore": BlockType.LAPIS_ORE,
    "minecraft:deepslate_copper_ore": BlockType.COPPER_ORE,
    # Filler blocks
    "minecraft:dirt": BlockType.DIRT,
    "minecraft:grass_block": BlockType.DIRT,
    "minecraft:coarse_dirt": BlockType.DIRT,
    "minecraft:rooted_dirt": BlockType.DIRT,
    "minecraft:podzol": BlockType.DIRT,
    "minecraft:mycelium": BlockType.DIRT,
    "minecraft:farmland": BlockType.DIRT,
    "minecraft:dirt_path": BlockType.DIRT,
    "minecraft:mud": BlockType.DIRT,
    "minecraft:gravel": BlockType.GRAVEL,
    "minecraft:granite": BlockType.GRANITE,
    "minecraft:polished_granite": BlockType.GRANITE,
    "minecraft:diorite": BlockType.DIORITE,
    "minecraft:polished_diorite": BlockType.DIORITE,
    "minecraft:andesite": BlockType.ANDESITE,
    "minecraft:polished_andesite": BlockType.ANDESITE,
    "minecraft:tuff": BlockType.TUFF,
    "minecraft:clay": BlockType.CLAY,
    # Sand → treat as dirt (close enough for mining sim)
    "minecraft:sand": BlockType.DIRT,
    "minecraft:red_sand": BlockType.DIRT,
    "minecraft:sandstone": BlockType.STONE,
    "minecraft:red_sandstone": BlockType.STONE,
    # Terracotta variants → stone
    "minecraft:terracotta": BlockType.STONE,
    "minecraft:white_terracotta": BlockType.STONE,
    "minecraft:orange_terracotta": BlockType.STONE,
    "minecraft:yellow_terracotta": BlockType.STONE,
    "minecraft:red_terracotta": BlockType.STONE,
    "minecraft:brown_terracotta": BlockType.STONE,
    "minecraft:light_gray_terracotta": BlockType.STONE,
    # Calcite, dripstone, etc.
    "minecraft:calcite": BlockType.STONE,
    "minecraft:dripstone_block": BlockType.STONE,
    "minecraft:pointed_dripstone": BlockType.STONE,
    "minecraft:amethyst_block": BlockType.STONE,
    "minecraft:budding_amethyst": BlockType.STONE,
    "minecraft:smooth_basalt": BlockType.STONE,
    "minecraft:basalt": BlockType.STONE,
    # Water/lava → treat as air (passable for sim purposes)
    "minecraft:water": BlockType.AIR,
    "minecraft:lava": BlockType.AIR,
    # Obsidian → stone (hard but solid)
    "minecraft:obsidian": BlockType.STONE,
    "minecraft:crying_obsidian": BlockType.STONE,
}

# Default for any block not in the mapping
DEFAULT_BLOCKTYPE: int = BlockType.STONE


def _normalize_ns(name: str) -> str:
    """Strip ``universal_`` prefix so Amulet universal names match too."""
    if name.startswith("universal_"):
        return name[len("universal_"):]
    return name


def mc_block_to_blocktype(namespaced_name: str) -> int:
    """Convert a Minecraft namespaced block name to a BlockType int.

    Handles both ``minecraft:`` and ``universal_minecraft:`` prefixes
    (the latter is used internally by Amulet Core).

    Returns ``DEFAULT_BLOCKTYPE`` (STONE) for unknown blocks.
    """
    result = MC_TO_BLOCKTYPE.get(namespaced_name)
    if result is not None:
        return result
    return MC_TO_BLOCKTYPE.get(_normalize_ns(namespaced_name), DEFAULT_BLOCKTYPE)


# ---------------------------------------------------------------------------
# Minecraft biome name → BiomeType
# ---------------------------------------------------------------------------

MC_BIOME_TO_BIOMETYPE: dict[str, int] = {
    # Plains-like (default category)
    "minecraft:plains": BiomeType.PLAINS,
    "minecraft:sunflower_plains": BiomeType.PLAINS,
    "minecraft:forest": BiomeType.PLAINS,
    "minecraft:flower_forest": BiomeType.PLAINS,
    "minecraft:birch_forest": BiomeType.PLAINS,
    "minecraft:dark_forest": BiomeType.PLAINS,
    "minecraft:old_growth_birch_forest": BiomeType.PLAINS,
    "minecraft:old_growth_pine_taiga": BiomeType.PLAINS,
    "minecraft:old_growth_spruce_taiga": BiomeType.PLAINS,
    "minecraft:taiga": BiomeType.PLAINS,
    "minecraft:snowy_taiga": BiomeType.PLAINS,
    "minecraft:snowy_plains": BiomeType.PLAINS,
    "minecraft:meadow": BiomeType.PLAINS,
    "minecraft:cherry_grove": BiomeType.PLAINS,
    "minecraft:swamp": BiomeType.PLAINS,
    "minecraft:mangrove_swamp": BiomeType.PLAINS,
    "minecraft:jungle": BiomeType.PLAINS,
    "minecraft:sparse_jungle": BiomeType.PLAINS,
    "minecraft:bamboo_jungle": BiomeType.PLAINS,
    "minecraft:savanna": BiomeType.PLAINS,
    "minecraft:savanna_plateau": BiomeType.PLAINS,
    "minecraft:windswept_savanna": BiomeType.PLAINS,
    "minecraft:beach": BiomeType.PLAINS,
    "minecraft:snowy_beach": BiomeType.PLAINS,
    "minecraft:stony_shore": BiomeType.PLAINS,
    "minecraft:river": BiomeType.PLAINS,
    "minecraft:frozen_river": BiomeType.PLAINS,
    "minecraft:ocean": BiomeType.PLAINS,
    "minecraft:deep_ocean": BiomeType.PLAINS,
    "minecraft:warm_ocean": BiomeType.PLAINS,
    "minecraft:lukewarm_ocean": BiomeType.PLAINS,
    "minecraft:cold_ocean": BiomeType.PLAINS,
    "minecraft:frozen_ocean": BiomeType.PLAINS,
    "minecraft:deep_lukewarm_ocean": BiomeType.PLAINS,
    "minecraft:deep_cold_ocean": BiomeType.PLAINS,
    "minecraft:deep_frozen_ocean": BiomeType.PLAINS,
    "minecraft:mushroom_fields": BiomeType.PLAINS,
    "minecraft:ice_spikes": BiomeType.PLAINS,
    "minecraft:desert": BiomeType.PLAINS,
    # Mountain-like
    "minecraft:windswept_hills": BiomeType.MOUNTAINS,
    "minecraft:windswept_gravelly_hills": BiomeType.MOUNTAINS,
    "minecraft:windswept_forest": BiomeType.MOUNTAINS,
    "minecraft:stony_peaks": BiomeType.MOUNTAINS,
    "minecraft:jagged_peaks": BiomeType.MOUNTAINS,
    "minecraft:frozen_peaks": BiomeType.MOUNTAINS,
    "minecraft:snowy_slopes": BiomeType.MOUNTAINS,
    "minecraft:grove": BiomeType.MOUNTAINS,
    # Badlands
    "minecraft:badlands": BiomeType.BADLANDS,
    "minecraft:eroded_badlands": BiomeType.BADLANDS,
    "minecraft:wooded_badlands": BiomeType.BADLANDS,
    # Cave biomes
    "minecraft:dripstone_caves": BiomeType.DRIPSTONE_CAVES,
    "minecraft:lush_caves": BiomeType.LUSH_CAVES,
    "minecraft:deep_dark": BiomeType.DRIPSTONE_CAVES,
}

DEFAULT_BIOME: int = BiomeType.PLAINS


def mc_biome_to_biometype(namespaced_name: str) -> int:
    """Convert a Minecraft namespaced biome name to a BiomeType int.

    Handles both ``minecraft:`` and ``universal_minecraft:`` prefixes.
    Returns ``DEFAULT_BIOME`` (PLAINS) for unknown biomes.
    """
    result = MC_BIOME_TO_BIOMETYPE.get(namespaced_name)
    if result is not None:
        return result
    return MC_BIOME_TO_BIOMETYPE.get(_normalize_ns(namespaced_name), DEFAULT_BIOME)
