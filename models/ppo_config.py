"""Environment factory and PPO model creation utilities.

Provides ``make_training_env`` for creating vectorised environments with
VecNormalize, and ``create_ppo_model`` for instantiating MaskablePPO with
the project's hyperparameters.
"""

from __future__ import annotations

from typing import Any, Callable

from prospect_rl.config import Config, NUM_ORE_TYPES, ORE_TYPE_CONFIGS
from prospect_rl.env.mining_env import MinecraftMiningEnv
from prospect_rl.models.policy_network import MiningFeatureExtractor
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def linear_schedule(
    initial_value: float,
    min_fraction: float = 0.1,
) -> Callable[[float], float]:
    """Linear decay from *initial_value* to a floor over training.

    SB3 calls the returned function with ``progress_remaining``
    which goes from 1.0 (start) to 0.0 (end).

    The floor is ``initial_value * min_fraction`` to prevent
    training stall at the end of long runs.
    """
    floor = initial_value * min_fraction

    def func(progress_remaining: float) -> float:
        return max(initial_value * progress_remaining, floor)
    return func


def make_training_env(
    n_envs: int = 4,
    stage_index: int = 0,
    seed: int = 42,
    world_class: type | None = None,
    cache_dir: str | None = None,
    real_fraction: float = 0.0,
    real_cache_dir: str | None = None,
    real_stage_index: int | None = None,
    use_subproc: bool = True,
) -> VecNormalize:
    """Create a vectorised, normalised training environment.

    Parameters
    ----------
    n_envs:
        Number of parallel environments.
    stage_index:
        Curriculum stage index (0-5).
    seed:
        Base random seed.  Each sub-env gets ``seed + i``.
    world_class:
        Override world class for all envs (e.g. ``RealChunkWorld``).
    cache_dir:
        Cache directory passed to ``MinecraftMiningEnv`` (for
        ``RealChunkWorld``).
    real_fraction:
        Fraction of envs to use ``RealChunkWorld`` (0.0 to 1.0).
        Requires ``real_cache_dir`` to be set.
    real_cache_dir:
        Cache directory for real chunk environments in mixed mode.
    real_stage_index:
        Curriculum stage for real chunk envs (defaults to
        ``stage_index``).
    use_subproc:
        Use ``SubprocVecEnv`` for multi-core parallelism (default).
        Set to ``False`` to use ``DummyVecEnv`` for debugging.

    Returns
    -------
    VecNormalize
        Wrapped vectorised environment.  Only the ``scalars`` key is
        normalised; ``voxels`` and ``pref`` are left untouched.
    """
    n_real = int(n_envs * real_fraction) if real_cache_dir else 0
    n_sim = n_envs - n_real
    real_stage = real_stage_index if real_stage_index is not None else stage_index

    def _make_sim(i: int):
        ore_idx = i % NUM_ORE_TYPES
        biome = ORE_TYPE_CONFIGS[ore_idx].forced_biome

        def _init():
            return MinecraftMiningEnv(
                curriculum_stage=stage_index,
                seed=seed + i,
                fixed_ore_index=ore_idx,
                forced_biome=biome,
                world_class=world_class,
                cache_dir=cache_dir,
            )
        return _init

    def _make_real(i: int):
        from prospect_rl.env.world.real_chunk_world import RealChunkWorld

        ore_idx = i % NUM_ORE_TYPES
        forced = ORE_TYPE_CONFIGS[ore_idx].forced_biome
        req_biome = int(forced) if forced is not None else None

        def _init():
            return MinecraftMiningEnv(
                curriculum_stage=real_stage,
                seed=seed + n_sim + i,
                fixed_ore_index=ore_idx,
                forced_biome=None,
                world_class=RealChunkWorld,
                cache_dir=real_cache_dir,
                required_biome=req_biome,
            )
        return _init

    env_fns = [_make_sim(i) for i in range(n_sim)]
    env_fns += [_make_real(i) for i in range(n_real)]

    if use_subproc and n_envs > 1:
        venv = SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        venv = DummyVecEnv(env_fns)
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
