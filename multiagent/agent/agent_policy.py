"""Multi-agent feature extractor extending the single-agent policy.

Extends ``MiningFeatureExtractor`` with:
- 16 voxel channels (added agent density channel)
- 80 scalar dimensions (+10: task/target/boundary features)
- Same FiLM+SE architecture, just wider inputs
"""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn

from prospect_rl.models.policy_network import (
    FiLMLayer,
    MiningFeatureExtractor,
    SqueezeExcitation3d,
)


class MultiAgentFeatureExtractor(MiningFeatureExtractor):
    """Feature extractor for multi-agent Dict(voxels, scalars, pref).

    Extends the single-agent extractor with:
    - 16 input channels (15 original + 1 agent density)
    - 80 scalar dims (70 original + 10 task/target/boundary)
    - Same output dimension structure

    The extra scalar dims are:
      [70:73]  rel_target_xyz (3) — normalized relative target position
      [73]     target_distance (1) — normalized distance to target
      [74:78]  task_type_onehot (4) — MOVE_TO/EXCAVATE/MINE_ORE/RETURN_TO
      [78]     distance_to_boundary (1) — normalized min distance to bbox edge
      [79]     inside_box_flag (1) — 1.0 if inside assigned bounding box
    """

    def __init__(
        self, observation_space: gym.spaces.Dict,
    ) -> None:
        # Call grandparent init to avoid MiningFeatureExtractor re-init
        # We need to set up everything ourselves with the wider dims
        voxel_shape = observation_space["voxels"].shape
        scalar_dim = observation_space["scalars"].shape[0]
        pref_dim = observation_space["pref"].shape[0]

        cnn_out_dim = 256
        scalar_out_dim = 64
        features_dim = cnn_out_dim + scalar_out_dim + pref_dim

        # Skip MiningFeatureExtractor.__init__, call BaseFeaturesExtractor
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        BaseFeaturesExtractor.__init__(
            self, observation_space, features_dim=features_dim,
        )

        in_channels = voxel_shape[0]  # 16
        conv_channels = [32, 64, 128]

        # 3D CNN layers
        self.conv1 = nn.Conv3d(
            in_channels, conv_channels[0],
            kernel_size=3, stride=2, padding=1,
        )
        self.conv2 = nn.Conv3d(
            conv_channels[0], conv_channels[1],
            kernel_size=3, stride=2, padding=1,
        )
        self.conv3 = nn.Conv3d(
            conv_channels[1], conv_channels[2],
            kernel_size=3, stride=2, padding=1,
        )

        # FiLM conditioning layers
        self.film1 = FiLMLayer(pref_dim, conv_channels[0])
        self.film2 = FiLMLayer(pref_dim, conv_channels[1])
        self.film3 = FiLMLayer(pref_dim, conv_channels[2])

        # Squeeze-Excitation
        self.se = SqueezeExcitation3d(conv_channels[2], reduction=4)
        self.relu = nn.ReLU()

        # Compute flattened CNN output size
        with torch.no_grad():
            dummy_voxel = torch.zeros(1, *voxel_shape)
            dummy_pref = torch.zeros(1, pref_dim)
            cnn_flat_size = self._cnn_forward(
                dummy_voxel, dummy_pref,
            ).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_flat_size, cnn_out_dim),
            nn.ReLU(),
        )

        # MLP for scalar features (wider for 80-dim input)
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, scalar_out_dim),
            nn.ReLU(),
        )
