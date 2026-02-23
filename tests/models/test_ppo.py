"""Tests for PPO integration (Phase 4)."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from prospect_rl.config import (
    NUM_ORE_TYPES,
    NUM_VOXEL_CHANNELS,
    OBS_WINDOW_X,
    OBS_WINDOW_Y,
    OBS_WINDOW_Z,
    SCALAR_OBS_DIM,
)


def _make_obs_space() -> spaces.Dict:
    """Create the observation space matching the current env."""
    return spaces.Dict({
        "voxels": spaces.Box(
            0, 1,
            (NUM_VOXEL_CHANNELS, OBS_WINDOW_Y, OBS_WINDOW_X, OBS_WINDOW_Z),
            np.float32,
        ),
        "scalars": spaces.Box(
            -np.inf, np.inf, (SCALAR_OBS_DIM,), np.float32,
        ),
        "pref": spaces.Box(
            0, 1, (NUM_ORE_TYPES,), np.float32,
        ),
    })


class TestFeatureExtractor:
    def test_output_dimension(self) -> None:
        from prospect_rl.models.policy_network import MiningFeatureExtractor

        obs_space = _make_obs_space()
        extractor = MiningFeatureExtractor(obs_space)
        # 128 (CNN) + 32 (scalar MLP) + 8 (pref) = 168
        assert extractor.features_dim == 168

    def test_forward_pass(self) -> None:
        from prospect_rl.models.policy_network import MiningFeatureExtractor

        obs_space = _make_obs_space()
        extractor = MiningFeatureExtractor(obs_space)

        obs = {
            "voxels": torch.rand(
                2, NUM_VOXEL_CHANNELS,
                OBS_WINDOW_Y, OBS_WINDOW_X, OBS_WINDOW_Z,
            ),
            "scalars": torch.randn(2, SCALAR_OBS_DIM),
            "pref": torch.rand(2, NUM_ORE_TYPES),
        }
        out = extractor(obs)
        assert out.shape == (2, 168)

    def test_pref_passed_through_unchanged(self) -> None:
        from prospect_rl.models.policy_network import MiningFeatureExtractor

        obs_space = _make_obs_space()
        extractor = MiningFeatureExtractor(obs_space)

        pref_data = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        pref_data[0] = 1.0
        pref = torch.tensor([pref_data], dtype=torch.float32)
        obs = {
            "voxels": torch.rand(
                1, NUM_VOXEL_CHANNELS,
                OBS_WINDOW_Y, OBS_WINDOW_X, OBS_WINDOW_Z,
            ),
            "scalars": torch.randn(1, SCALAR_OBS_DIM),
            "pref": pref,
        }
        out = extractor(obs)
        # Last NUM_ORE_TYPES features should be the pref vector
        np.testing.assert_array_almost_equal(
            out[0, -NUM_ORE_TYPES:].detach().numpy(),
            pref[0].numpy(),
        )


class TestPPOModel:
    def test_model_creates(self) -> None:
        from prospect_rl.models.ppo_config import (
            create_ppo_model,
            make_training_env,
        )

        env = make_training_env(n_envs=1, stage_index=0, seed=42)
        model = create_ppo_model(env, seed=42)
        assert model is not None

    def test_vecnormalize_scalars_only(self) -> None:
        from prospect_rl.models.ppo_config import make_training_env

        env = make_training_env(n_envs=1, stage_index=0, seed=42)

        # Step a few times to collect running stats
        obs = env.reset()
        for _ in range(10):
            action = [env.action_space.sample()]
            obs, _, _, _ = env.step(action)

        # Pref should be unchanged (values between 0-1, sum to 1)
        pref = obs["pref"][0]
        assert np.all(pref >= 0)
        assert np.all(pref <= 1)
        np.testing.assert_almost_equal(pref.sum(), 1.0, decimal=4)

        # Voxels should still be binary (0 or 1)
        voxels = obs["voxels"][0]
        unique_vals = np.unique(voxels)
        for v in unique_vals:
            assert v in (0.0, 1.0), f"Voxel value {v} should be binary"

    def test_short_training_no_crash(self) -> None:
        from prospect_rl.models.ppo_config import (
            create_ppo_model,
            make_training_env,
        )

        env = make_training_env(n_envs=2, stage_index=0, seed=42)
        model = create_ppo_model(
            env, seed=42,
            n_steps=64, batch_size=32,
        )
        model.learn(total_timesteps=256)

    def test_predict_returns_valid_action(self) -> None:
        from prospect_rl.models.ppo_config import (
            create_ppo_model,
            make_training_env,
        )

        env = make_training_env(n_envs=1, stage_index=0, seed=42)
        model = create_ppo_model(
            env, seed=42,
            n_steps=64, batch_size=32,
        )
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert 0 <= int(action[0]) < 9
