"""OpenSimplex noise wrappers with fractal Brownian motion (fBm).

All functions are vectorized via ``opensimplex.noise3array`` / ``noise2array``.
"""

from __future__ import annotations

import numpy as np
import opensimplex


def octave_noise_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    seed: int = 0,
    scale: float = 1.0,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Generate 3D fractal Brownian motion noise over coordinate arrays.

    Parameters
    ----------
    x, y, z:
        1-D coordinate arrays.  The returned array has shape
        ``(z.size, y.size, x.size)``.
    seed:
        Random seed for the noise generator.
    scale:
        Base spatial scale (higher = smoother).
    octaves:
        Number of noise layers.
    persistence:
        Amplitude multiplier per octave.
    lacunarity:
        Frequency multiplier per octave.

    Returns
    -------
    np.ndarray of shape ``(z.size, y.size, x.size)`` with values roughly in
    ``[-1, 1]``.
    """
    opensimplex.seed(seed)

    xs = x.astype(np.float64) / scale
    ys = y.astype(np.float64) / scale
    zs = z.astype(np.float64) / scale

    result = np.zeros((zs.size, ys.size, xs.size), dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        result += amplitude * opensimplex.noise3array(
            xs * frequency, ys * frequency, zs * frequency
        )
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normalise so the output stays in approximately [-1, 1]
    result /= max_amplitude
    return result


def octave_noise_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
    scale: float = 1.0,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Generate 2D fractal Brownian motion noise over coordinate arrays.

    Parameters
    ----------
    x, y:
        1-D coordinate arrays.  The returned array has shape
        ``(y.size, x.size)``.

    Returns
    -------
    np.ndarray of shape ``(y.size, x.size)`` with values roughly in
    ``[-1, 1]``.
    """
    opensimplex.seed(seed)

    xs = x.astype(np.float64) / scale
    ys = y.astype(np.float64) / scale

    result = np.zeros((ys.size, xs.size), dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        result += amplitude * opensimplex.noise2array(
            xs * frequency, ys * frequency
        )
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    result /= max_amplitude
    return result
