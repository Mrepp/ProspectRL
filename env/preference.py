"""Preference vector sampling and reward scalarization.

During training a random preference vector is sampled at each episode reset.
The scalarize method combines the vectorized ore/cost rewards into a single
scalar reward used by PPO.
"""

from __future__ import annotations

import numpy as np
from prospect_rl.config import NUM_ORE_TYPES, REWARD_ALPHA

# Fixed cost weight vector — order matches r_cost from compute_reward_vector
_COST_WEIGHT_VEC = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)


class PreferenceManager:
    """Samples and scalarizes preference-weighted rewards."""

    def __init__(self, num_ores: int = NUM_ORE_TYPES, seed: int | None = None) -> None:
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
            indices = self.rng.choice(self.num_ores, size=2, replace=False)
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
        w_ore: np.ndarray,
        r_ore: np.ndarray,
        r_cost: np.ndarray,
    ) -> float:
        """Combine ore-reward and cost vectors into a scalar reward.

        ``R = REWARD_ALPHA * dot(w_ore, r_ore) + sum(r_cost)``

        ``r_cost`` values already contain their signs (negative), so we
        simply sum them.
        """
        ore_reward = float(REWARD_ALPHA * np.dot(w_ore, r_ore))
        cost_reward = float(np.sum(r_cost))
        return ore_reward + cost_reward
