"""Custom CNN+MLP feature extractor for the mining environment.

Processes the Dict observation space (``voxels`` + ``scalars`` + ``pref``)
into a single feature vector for MaskablePPO.  The voxels branch uses 3D
convolutions, the scalars branch uses a small MLP, and pref is passed
through raw.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MiningFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for Dict(voxels, scalars, pref) observations.

    Architecture::

        voxels (13,32,9,9) -> Conv3d pipeline -> FC -> [128]
        scalars (17)        -> MLP(64->32)     -> [32]
        pref (8)            -> identity        -> [  8]
                                          concat -> [168]
    """

    def __init__(
        self, observation_space: gym.spaces.Dict,
    ) -> None:
        voxel_shape = observation_space["voxels"].shape
        scalar_dim = observation_space["scalars"].shape[0]
        pref_dim = observation_space["pref"].shape[0]

        cnn_out_dim = 128
        scalar_out_dim = 32
        features_dim = cnn_out_dim + scalar_out_dim + pref_dim
        super().__init__(
            observation_space, features_dim=features_dim,
        )

        in_channels = voxel_shape[0]

        # 3D CNN for voxel processing
        self.voxel_cnn = nn.Sequential(
            nn.Conv3d(
                in_channels, 32,
                kernel_size=3, stride=2, padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                32, 64, kernel_size=3, stride=2, padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                64, 64, kernel_size=3, stride=2, padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened CNN output size via dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, *voxel_shape)
            cnn_flat_size = self.voxel_cnn(dummy).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_flat_size, cnn_out_dim),
            nn.ReLU(),
        )

        # MLP for scalar features
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, scalar_out_dim),
            nn.ReLU(),
        )

    def forward(
        self, observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        voxel_features = self.cnn_fc(
            self.voxel_cnn(observations["voxels"]),
        )
        scalar_features = self.scalar_net(
            observations["scalars"],
        )
        return torch.cat(
            [
                voxel_features,
                scalar_features,
                observations["pref"],
            ],
            dim=1,
        )
