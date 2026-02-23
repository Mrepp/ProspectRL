"""3D cave generation using interpolated coarse noise.

Caves are carved by setting blocks to AIR wherever a 3D noise field exceeds
a configurable threshold.  Bedrock (y=0) is never carved.

Uses coarse-resolution random noise with trilinear interpolation via
``scipy.ndimage.zoom`` for smooth caves without expensive per-voxel simplex
noise.  All operations are numpy-vectorized.
"""

from __future__ import annotations

import numpy as np
from prospect_rl.config import BlockType, CaveConfig
from scipy.ndimage import zoom


def _coarse_fbm(
    sx: int,
    sy: int,
    sz: int,
    seed: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
) -> np.ndarray:
    """Generate a smooth 3D noise field using coarse noise + zoom.

    For each octave, generate random values at a coarse resolution
    (determined by ``scale / frequency``) and upscale to ``(sx, sy, sz)``
    via trilinear interpolation.  Sum octaves with decreasing amplitude.
    """
    result = np.zeros((sx, sy, sz), dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_amp = 0.0

    for oct_i in range(octaves):
        cell = max(1.0, scale / frequency)
        # Coarse grid dimensions (at least 2 along each axis)
        cx = max(2, int(np.ceil(sx / cell)) + 1)
        cy = max(2, int(np.ceil(sy / cell)) + 1)
        cz = max(2, int(np.ceil(sz / cell)) + 1)

        rng = np.random.default_rng(seed + oct_i * 31337)
        coarse = rng.standard_normal((cx, cy, cz))

        # Upscale to world size with smooth interpolation
        factors = (sx / cx, sy / cy, sz / cz)
        upscaled = zoom(coarse, factors, order=1)
        # Trim to exact size (zoom may overshoot by 1)
        upscaled = upscaled[:sx, :sy, :sz]

        result += amplitude * upscaled
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    result /= max_amp
    return result


class CaveGenerator:
    """Carves caves into a 3D world grid using vectorized noise thresholding."""

    @staticmethod
    def carve_caves(
        world_blocks: np.ndarray,
        size: tuple[int, int, int],
        seed: int,
        config: CaveConfig,
    ) -> np.ndarray:
        """Carve caves into *world_blocks* in-place.

        Uses fBm from coarse noise + trilinear interpolation.
        """
        sx, sy, sz = size
        cave_seed = seed + 99999

        noise_field = _coarse_fbm(
            sx, sy, sz,
            seed=cave_seed,
            scale=config.noise_scale,
            octaves=config.octaves,
            persistence=config.persistence,
            lacunarity=config.lacunarity,
        )

        cave_mask = noise_field > config.threshold
        cave_mask[:, 0, :] = False  # protect bedrock

        world_blocks[cave_mask] = np.int8(BlockType.AIR)
        return world_blocks
