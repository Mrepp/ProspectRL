"""REINFORCE trainer for the CoordinatorGNN.

Trains the GNN using REINFORCE with baseline on team-level reward.
Operates at the coordinator's replan timescale (every K steps).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class CoordinatorTrainer:
    """REINFORCE trainer for the coordinator GNN.

    Parameters
    ----------
    gnn:
        The CoordinatorGNN (or CoordinatorGNNSimple) module.
    lr:
        Learning rate.
    ent_coef:
        Entropy bonus coefficient.
    baseline_window:
        Number of replan cycles for EMA baseline.
    """

    def __init__(
        self,
        gnn: torch.nn.Module,
        lr: float = 3e-4,
        ent_coef: float = 0.01,
        baseline_window: int = 100,
    ) -> None:
        self._gnn = gnn
        self._optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
        self._ent_coef = ent_coef

        # EMA baseline for variance reduction
        self._ema_alpha: float = 2.0 / (baseline_window + 1)
        self._baseline: float = 0.0

        # Logging
        self._total_updates: int = 0
        self._last_loss: float = 0.0
        self._last_entropy: float = 0.0

    def update(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        team_reward: float,
    ) -> dict[str, float]:
        """Perform one REINFORCE update.

        Parameters
        ----------
        x_dict:
            Node features (agent and region).
        edge_index_dict:
            Edge indices for the heterogeneous graph.
        row_idx:
            Agent indices from Hungarian matching.
        col_idx:
            Region indices from Hungarian matching.
        team_reward:
            Total team reward over the K-step window since last replan.

        Returns
        -------
        Dict with training metrics.
        """
        # Update EMA baseline
        if self._total_updates == 0:
            self._baseline = team_reward
        else:
            self._baseline = (
                self._ema_alpha * team_reward
                + (1.0 - self._ema_alpha) * self._baseline
            )
        advantage = team_reward - self._baseline

        # Guard: no valid assignments → skip update to prevent NaN loss
        if len(row_idx) == 0:
            self._total_updates += 1
            return {
                "coordinator/loss": 0.0,
                "coordinator/policy_loss": 0.0,
                "coordinator/entropy": 0.0,
                "coordinator/advantage": float(advantage),
                "coordinator/baseline": self._baseline,
                "coordinator/team_reward": team_reward,
            }

        # Ensure inputs are on the same device as GNN
        device = next(self._gnn.parameters()).device
        x_dict = {k: v.to(device) for k, v in x_dict.items()}

        # Forward pass
        scores = self._gnn(x_dict, edge_index_dict)  # (N_a, N_r)

        # Compute log-probabilities of selected assignments
        log_probs = F.log_softmax(scores, dim=1)

        # Gather log-probs for the Hungarian-matched pairs
        row_t = torch.tensor(row_idx, dtype=torch.long, device=scores.device)
        col_t = torch.tensor(col_idx, dtype=torch.long, device=scores.device)
        selected_log_probs = log_probs[row_t, col_t]

        # REINFORCE loss: -log_prob * advantage
        policy_loss = -(selected_log_probs * advantage).mean()

        # Entropy bonus: encourage exploration in score distribution
        probs = F.softmax(scores, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_loss = -self._ent_coef * entropy

        loss = policy_loss + entropy_loss

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._gnn.parameters(), 1.0)
        self._optimizer.step()

        self._total_updates += 1
        self._last_loss = float(loss.item())
        self._last_entropy = float(entropy.item())

        return {
            "coordinator/loss": float(loss.item()),
            "coordinator/policy_loss": float(policy_loss.item()),
            "coordinator/entropy": float(entropy.item()),
            "coordinator/advantage": float(advantage),
            "coordinator/baseline": self._baseline,
            "coordinator/team_reward": team_reward,
        }

    @property
    def total_updates(self) -> int:
        return self._total_updates
