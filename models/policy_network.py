"""Custom CNN+MLP feature extractor for the mining environment.

Processes the Dict observation space (``voxels`` + ``scalars`` + ``pref``)
into a single feature vector for MaskablePPO.  The voxels branch uses 3D
convolutions with FiLM conditioning and squeeze-excitation attention,
the scalars branch uses a small MLP, and pref is passed through raw.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SqueezeExcitation3d(nn.Module):
    """Squeeze-Excitation block for 3D feature maps.

    GlobalAvgPool3d -> FC(C->C//r) -> ReLU -> FC(C//r->C) -> Sigmoid -> scale.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        scale = self.squeeze(x).view(b, c)
        scale = self.excitation(scale).view(b, c, 1, 1, 1)
        return x * scale


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Projects a conditioning vector to per-channel affine parameters
    and applies: out = gamma * features + beta.

    Initialized near-identity (gamma~1, beta~0) for stable training.
    """

    def __init__(self, cond_dim: int, num_channels: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, num_channels)
        self.beta_proj = nn.Linear(cond_dim, num_channels)

        # Near-identity init: gamma ≈ 1.0, beta ≈ 0.0
        nn.init.normal_(self.gamma_proj.weight, std=0.01)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.normal_(self.beta_proj.weight, std=0.01)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self, features: torch.Tensor, conditioning: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self.gamma_proj(conditioning).view(
            conditioning.shape[0], -1, 1, 1, 1,
        )
        beta = self.beta_proj(conditioning).view(
            conditioning.shape[0], -1, 1, 1, 1,
        )
        return gamma * features + beta


class MiningFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for Dict(voxels, scalars, pref) observations.

    Architecture::

        pref (8) ─── FiLM conditioning ──→ injected into each conv layer
                                      └──→ identity → [8]
        voxels (15,11,7,7) → Conv3d(15→32)  → FiLM → ReLU
                           → Conv3d(32→64)  → FiLM → ReLU
                           → Conv3d(64→128) → FiLM → ReLU
                           → SE(128, r=4)
                           → Flatten → FC(→256) → ReLU       → [256]
        scalars (70)       → MLP(128→64)                     → [ 64]
                                                        concat → [328]
    """

    def __init__(
        self, observation_space: gym.spaces.Dict,
    ) -> None:
        voxel_shape = observation_space["voxels"].shape
        scalar_dim = observation_space["scalars"].shape[0]
        pref_dim = observation_space["pref"].shape[0]

        cnn_out_dim = 256
        scalar_out_dim = 64
        features_dim = cnn_out_dim + scalar_out_dim + pref_dim
        super().__init__(
            observation_space, features_dim=features_dim,
        )

        in_channels = voxel_shape[0]
        conv_channels = [32, 64, 128]

        # 3D CNN layers (FiLM applied between conv and activation)
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

        # Squeeze-Excitation after last conv
        self.se = SqueezeExcitation3d(conv_channels[2], reduction=4)

        self.relu = nn.ReLU()

        # Compute flattened CNN output size via dummy forward
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

        # MLP for scalar features (wider for 70-dim fog-of-war input)
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, scalar_out_dim),
            nn.ReLU(),
        )

    def _cnn_forward(
        self, voxels: torch.Tensor, pref: torch.Tensor,
    ) -> torch.Tensor:
        """Run voxels through Conv+FiLM+ReLU+SE pipeline, return flattened."""
        x = self.relu(self.film1(self.conv1(voxels), pref))
        x = self.relu(self.film2(self.conv2(x), pref))
        x = self.relu(self.film3(self.conv3(x), pref))
        x = self.se(x)
        return x.flatten(1)

    def forward(
        self, observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pref = observations["pref"]
        # Upcast float16 voxel obs to float32 for Conv3d compatibility
        voxels = observations["voxels"].float()
        voxel_features = self.cnn_fc(
            self._cnn_forward(voxels, pref),
        )
        scalar_features = self.scalar_net(
            observations["scalars"],
        )
        return torch.cat(
            [voxel_features, scalar_features, pref],
            dim=1,
        )
