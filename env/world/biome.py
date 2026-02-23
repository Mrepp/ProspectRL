"""Biome map generation using 2D OpenSimplex noise.

Generates a column-based biome map (same biome for entire Y column)
using temperature/humidity noise layers, matching Minecraft's approach.
All operations are numpy-vectorized.
"""

from __future__ import annotations

import numpy as np
from prospect_rl.config import BiomeType
from prospect_rl.env.world.noise_utils import octave_noise_2d


class BiomeGenerator:
    """Generates 2D biome maps from OpenSimplex noise."""

    @staticmethod
    def generate_biome_map(sx: int, sz: int, seed: int) -> np.ndarray:
        """Generate a 2D biome map of shape ``(sx, sz)``.

        Uses two noise layers (temperature and humidity) to assign biomes:
        - Low temperature -> Mountains
        - High temperature, low humidity -> Badlands
        - Mid temperature, high humidity -> Dripstone Caves / Lush Caves
        - Everything else -> Plains (most common)

        Parameters
        ----------
        sx, sz:
            Horizontal dimensions of the world.
        seed:
            Random seed for noise generation.

        Returns
        -------
        np.ndarray of shape ``(sx, sz)`` with ``int8`` BiomeType values.
        """
        x_coords = np.arange(sx, dtype=np.float64)
        z_coords = np.arange(sz, dtype=np.float64)

        # Temperature noise — large scale for smooth biome regions
        temp_noise = octave_noise_2d(
            x_coords, z_coords,
            seed=seed, scale=50.0, octaves=2,
            persistence=0.5, lacunarity=2.0,
        )
        # noise2d returns shape (z, x) — transpose to (x, z)
        temp_noise = temp_noise.T

        # Humidity noise — offset seed for independence
        humid_noise = octave_noise_2d(
            x_coords, z_coords,
            seed=seed + 7777, scale=50.0, octaves=2,
            persistence=0.5, lacunarity=2.0,
        )
        humid_noise = humid_noise.T

        # Default everything to Plains
        biome_map = np.full((sx, sz), BiomeType.PLAINS, dtype=np.int8)

        # Mountains: low temperature (< -0.3)
        mountains_mask = temp_noise < -0.3
        biome_map[mountains_mask] = BiomeType.MOUNTAINS

        # Badlands: high temperature (> 0.4) AND low humidity (< 0.0)
        badlands_mask = (temp_noise > 0.4) & (humid_noise < 0.0)
        biome_map[badlands_mask] = BiomeType.BADLANDS

        # Dripstone Caves: mid temp (-0.1 to 0.3) AND high humidity (> 0.3)
        dripstone_mask = (
            (temp_noise > -0.1) & (temp_noise < 0.3) & (humid_noise > 0.3)
        )
        biome_map[dripstone_mask] = BiomeType.DRIPSTONE_CAVES

        # Lush Caves: mid temp (0.0 to 0.4) AND very high humidity (> 0.5)
        lush_mask = (
            (temp_noise > 0.0) & (temp_noise < 0.4) & (humid_noise > 0.5)
        )
        biome_map[lush_mask] = BiomeType.LUSH_CAVES

        return biome_map
