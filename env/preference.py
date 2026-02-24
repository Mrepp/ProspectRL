"""Preference vector sampling and reward scalarization.

During training a random preference vector is sampled at each episode reset.
The scalarize method combines the reward components into a single scalar
reward used by PPO.
"""

from __future__ import annotations

import numpy as np
from prospect_rl.config import NUM_ORE_TYPES, RewardConfig

_DEFAULT_CFG = RewardConfig()


class PreferenceManager:
    """Samples and scalarizes preference-weighted rewards."""

    def __init__(
        self,
        num_ores: int = NUM_ORE_TYPES,
        seed: int | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.num_ores = num_ores

    def sample(self, mode: str = "one_hot") -> np.ndarray:
        """Sample a preference vector of shape ``(num_ores,)`` summing to 1.0.

        Parameters
        ----------
        mode:
            ``"one_hot"`` — single random ore gets weight 1.0.
            ``"two_mix"`` — two random ores share the weight.
            ``"dirichlet"`` — Dirichlet(alpha=0.5) sample.
        """
        if mode == "one_hot":
            w = np.zeros(self.num_ores, dtype=np.float32)
            w[self.rng.integers(self.num_ores)] = 1.0
            return w

        if mode == "two_mix":
            w = np.zeros(self.num_ores, dtype=np.float32)
            indices = self.rng.choice(
                self.num_ores, size=2, replace=False,
            )
            split = self.rng.uniform(0.1, 0.9)
            w[indices[0]] = np.float32(split)
            w[indices[1]] = np.float32(1.0 - split)
            return w

        if mode == "dirichlet":
            alpha = np.full(self.num_ores, 0.5, dtype=np.float64)
            w = self.rng.dirichlet(alpha).astype(np.float32)
            # Ensure exact sum to 1.0 after float32 cast
            w /= w.sum()
            return w

        raise ValueError(f"Unknown preference mode: {mode!r}")

    @staticmethod
    def scalarize(
        r_harvest: float,
        r_adjacent: float,
        r_clear: float,
        r_ops: float,
        reward_config: RewardConfig | None = None,
    ) -> float:
        """Combine reward components into a scalar reward.

        ``R = ALPHA * r_harvest + r_adjacent + r_clear + r_ops``

        ``r_adjacent`` already contains the ``-BETA`` factor.
        ``r_ops`` values already contain their signs (negative).
        """
        cfg = reward_config or _DEFAULT_CFG
        return (
            cfg.harvest_alpha * r_harvest
            + r_adjacent
            + r_clear
            + r_ops
        )

    @staticmethod
    def compute_episode_bonus(
        potential: float,
        reward_config: RewardConfig | None = None,
    ) -> float:
        """End-of-episode bonus proportional to final harvest potential."""
        cfg = reward_config or _DEFAULT_CFG
        return cfg.episode_bonus_gamma * potential
