"""Environment factory and PPO model creation utilities.

Provides ``make_training_env`` for creating vectorised environments with
VecNormalize, and ``create_ppo_model`` for instantiating MaskablePPO with
the project's hyperparameters.
"""

from __future__ import annotations

from typing import Any, Callable

from prospect_rl.config import Config
from prospect_rl.env.mining_env import MinecraftMiningEnv
from prospect_rl.models.policy_network import MiningFeatureExtractor
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def linear_schedule(
    initial_value: float,
) -> Callable[[float], float]:
    """Linear decay from *initial_value* to 0 over training.

    SB3 calls the returned function with ``progress_remaining``
    which goes from 1.0 (start) to 0.0 (end).
    """
    def func(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    return func


def make_training_env(
    n_envs: int = 4,
    stage_index: int = 0,
    seed: int = 42,
) -> VecNormalize:
    """Create a vectorised, normalised training environment.

    Parameters
    ----------
    n_envs:
        Number of parallel environments.
    stage_index:
        Curriculum stage index (0-4).
    seed:
        Base random seed.  Each sub-env gets ``seed + i``.

    Returns
    -------
    VecNormalize
        Wrapped vectorised environment.  Only the ``scalars`` key is
        normalised; ``voxels`` and ``pref`` are left untouched.
    """

    def _make(i: int):
        def _init():
            return MinecraftMiningEnv(
                curriculum_stage=stage_index,
                seed=seed + i,
            )
        return _init

    venv = DummyVecEnv([_make(i) for i in range(n_envs)])
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_obs_keys=["scalars"],
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )


def create_ppo_model(
    env: VecNormalize,
    config: Config | None = None,
    tensorboard_log: str | None = None,
    seed: int = 42,
    **overrides: Any,
) -> MaskablePPO:
    """Instantiate a MaskablePPO model with project hyperparameters.

    Parameters
    ----------
    env:
        Vectorised training environment (should be VecNormalize-wrapped).
    config:
        Project configuration.  Defaults to ``Config()``.
    tensorboard_log:
        Path for TensorBoard logs.
    seed:
        Random seed for the model.
    **overrides:
        Override any PPO hyperparameter by name.
    """
    if config is None:
        config = Config()

    ppo = config.ppo
    kwargs: dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "env": env,
        "learning_rate": linear_schedule(ppo.learning_rate),
        "n_steps": ppo.n_steps,
        "batch_size": ppo.batch_size,
        "n_epochs": ppo.n_epochs,
        "gamma": ppo.gamma,
        "gae_lambda": ppo.gae_lambda,
        "clip_range": ppo.clip_range,
        "ent_coef": ppo.ent_coef,
        "vf_coef": ppo.vf_coef,
        "max_grad_norm": ppo.max_grad_norm,
        "normalize_advantage": ppo.normalize_advantage,
        "policy_kwargs": {
            "features_extractor_class": MiningFeatureExtractor,
            "net_arch": {
                "pi": ppo.pi_net_arch,
                "vf": ppo.vf_net_arch,
            },
        },
        "seed": seed,
        "verbose": 1,
    }

    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = tensorboard_log

    kwargs.update(overrides)
    return MaskablePPO(**kwargs)
