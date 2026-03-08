"""Heterogeneous GNN for coordinator score prediction.

Uses ``torch_geometric`` heterogeneous graph support with GATv2Conv
message passing. Produces per-(agent, region) utility scores for
Hungarian matching.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, HeteroConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# Agent node features: position(3) + fuel(1) + inventory(8) +
#   current_assignment(3) + steps_since_replan(1) + preference(8) = 24
AGENT_FEATURE_DIM = 24

# Region node features: center(3) + expected_remaining(8) + info_gain(1) +
#   explored_frac(1) + biome_onehot(5) + assigned_agents(1) +
#   bounding_box_mask(1) = 20
REGION_FEATURE_DIM = 20


class CoordinatorGNN(nn.Module):
    """Heterogeneous GNN for agent-to-region score prediction.

    Builds a bipartite graph with agent and region nodes, uses
    GATv2Conv for message passing, and outputs a score matrix
    for Hungarian matching.

    Parameters
    ----------
    agent_dim:
        Input dimension for agent nodes.
    region_dim:
        Input dimension for region nodes.
    hidden:
        Hidden dimension for all layers.
    layers:
        Number of message passing rounds.
    heads:
        Number of attention heads in GATv2Conv.
    """

    def __init__(
        self,
        agent_dim: int = AGENT_FEATURE_DIM,
        region_dim: int = REGION_FEATURE_DIM,
        hidden: int = 128,
        layers: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()

        if not _HAS_PYG:
            raise ImportError(
                "torch_geometric is required for CoordinatorGNN. "
                "Install with: pip install torch-geometric"
            )

        self.agent_proj = nn.Linear(agent_dim, hidden)
        self.region_proj = nn.Linear(region_dim, hidden)

        head_dim = hidden // heads

        self.convs = nn.ModuleList()
        for _ in range(layers):
            conv = HeteroConv({
                ("agent", "considers", "region"): GATv2Conv(
                    hidden, head_dim, heads=heads, add_self_loops=False,
                ),
                ("region", "considered_by", "agent"): GATv2Conv(
                    hidden, head_dim, heads=heads, add_self_loops=False,
                ),
                ("agent", "near", "agent"): GATv2Conv(
                    hidden, head_dim, heads=heads,
                ),
                ("region", "adjacent", "region"): GATv2Conv(
                    hidden, head_dim, heads=heads,
                ),
            }, aggr="sum")
            self.convs.append(conv)

        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Forward pass producing score matrix.

        Parameters
        ----------
        x_dict:
            Node features: ``{"agent": (N_a, agent_dim), "region": (N_r, region_dim)}``.
        edge_index_dict:
            Edge indices for each edge type.

        Returns
        -------
        scores:
            Shape ``(N_agents, N_regions)`` utility scores.
        """
        h = {
            "agent": F.relu(self.agent_proj(x_dict["agent"])),
            "region": F.relu(self.region_proj(x_dict["region"])),
        }

        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            h = {k: F.relu(v) for k, v in h_new.items()}

        agent_emb = h["agent"]    # (N_a, hidden)
        region_emb = h["region"]  # (N_r, hidden)

        n_agents = agent_emb.size(0)
        n_regions = region_emb.size(0)

        # All pairs: outer product
        a_exp = agent_emb.unsqueeze(1).expand(n_agents, n_regions, -1)
        r_exp = region_emb.unsqueeze(0).expand(n_agents, n_regions, -1)
        pair_features = torch.cat([a_exp, r_exp], dim=-1)

        scores = self.score_head(pair_features).squeeze(-1)  # (N_a, N_r)
        return scores


class CoordinatorGNNSimple(nn.Module):
    """Simplified coordinator for use without torch_geometric.

    Uses standard linear layers for agent-region scoring.
    Suitable for testing and environments without PyG installed.
    """

    def __init__(
        self,
        agent_dim: int = AGENT_FEATURE_DIM,
        region_dim: int = REGION_FEATURE_DIM,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.agent_proj = nn.Sequential(
            nn.Linear(agent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.region_proj = nn.Sequential(
            nn.Linear(region_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict | None = None) -> torch.Tensor:
        agent_emb = self.agent_proj(x_dict["agent"])
        region_emb = self.region_proj(x_dict["region"])

        n_agents = agent_emb.size(0)
        n_regions = region_emb.size(0)

        a_exp = agent_emb.unsqueeze(1).expand(n_agents, n_regions, -1)
        r_exp = region_emb.unsqueeze(0).expand(n_agents, n_regions, -1)
        pair_features = torch.cat([a_exp, r_exp], dim=-1)

        scores = self.score_head(pair_features).squeeze(-1)
        return scores
