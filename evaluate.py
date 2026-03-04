"""Evaluate a trained model across a preference grid.

Usage::

    python -m prospect_rl.evaluate --model ./checkpoints/final/model.zip
    python -m prospect_rl.evaluate --model ./checkpoints/final/model.zip --episodes 20

    # Evaluate on real Minecraft chunks:
    python -m prospect_rl.evaluate --model ./checkpoints/final/model.zip \\
        --cache-dir data/chunk_cache/combined/ --episodes 20
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from prospect_rl.config import NUM_ORE_TYPES, ORE_TYPE_CONFIGS, ORE_TYPES

# Stage index for the real-chunk Stage 1 evaluation config
_STAGE1_REAL_EVAL_INDEX = 6


def _build_preference_grid() -> list[tuple[str, np.ndarray]]:
    """Build evaluation preference vectors: 7 one-hot + 3 mixed."""
    ore_names = [
        "coal", "iron", "gold", "diamond",
        "redstone", "emerald", "lapis",
    ]
    grid: list[tuple[str, np.ndarray]] = []

    # One-hot preferences
    for i, name in enumerate(ore_names):
        pref = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        pref[i] = 1.0
        grid.append((f"one_hot_{name}", pref))

    # Mixed preferences
    mixed = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
    mixed[1] = 0.5  # iron
    mixed[3] = 0.5  # diamond
    grid.append(("mix_iron_diamond", mixed.copy()))

    mixed2 = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
    mixed2[0] = 0.33  # coal
    mixed2[1] = 0.33  # iron
    mixed2[2] = 0.34  # gold
    grid.append(("mix_coal_iron_gold", mixed2.copy()))

    uniform = np.full(NUM_ORE_TYPES, 1.0 / NUM_ORE_TYPES, dtype=np.float32)
    grid.append(("uniform", uniform))

    return grid


def _load_obs_normalizer(
    vecnormalize_path: str,
    stage_index: int,
) -> callable:
    """Load VecNormalize stats and return a function that normalizes obs.

    The returned callable applies the same scalar normalization that was
    used during training (zero-mean, unit-variance, clipped) so the
    policy sees the distribution it was trained on.

    *stage_index* should be the **training** stage (not the eval stage)
    so the dummy env produces observations with the right shape.
    """
    from prospect_rl.env.mining_env import MinecraftMiningEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    dummy = DummyVecEnv([
        lambda: MinecraftMiningEnv(curriculum_stage=stage_index, seed=0),
    ])
    vn = VecNormalize.load(vecnormalize_path, dummy)
    vn.training = False
    vn.norm_reward = False

    # Extract running statistics for the "scalars" key
    obs_rms = vn.obs_rms
    clip_obs = vn.clip_obs

    if isinstance(obs_rms, dict):
        rms = obs_rms["scalars"]
    else:
        rms = obs_rms

    mean = rms.mean.astype(np.float32)
    std = np.sqrt(rms.var.astype(np.float32) + 1e-8)

    def normalize(obs: dict) -> dict:
        out = dict(obs)
        out["scalars"] = np.clip(
            (obs["scalars"] - mean) / std,
            -clip_obs,
            clip_obs,
        ).astype(np.float32)
        return out

    return normalize


def evaluate(
    model_path: str,
    vecnormalize_path: str | None = None,
    stage_index: int = 0,
    n_episodes: int = 10,
    cache_dir: str | None = None,
    min_ores: int = 1,
) -> dict:
    """Run evaluation across the preference grid.

    Parameters
    ----------
    cache_dir:
        Path to chunk cache directory for real chunk evaluation.
        When set, auto-selects ``stage1_real_eval`` if stage is 0.
    min_ores:
        Minimum ore count per chunk (filters barren chunks).

    Returns a dict mapping preference name to per-episode metrics.
    """
    from prospect_rl.env.mining_env import MinecraftMiningEnv
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path)

    # Determine world class and effective eval stage
    world_class = None
    eval_stage = stage_index
    if cache_dir is not None:
        from prospect_rl.env.world.real_chunk_world import RealChunkWorld
        world_class = RealChunkWorld
        if stage_index == 0:
            eval_stage = _STAGE1_REAL_EVAL_INDEX
            print(
                f"Auto-selected stage1_real_eval (stage {eval_stage}) "
                "for real chunks with Stage 1 rewards"
            )

    # Load VecNormalize stats using the TRAINING stage
    normalize = None
    if vecnormalize_path is not None:
        normalize = _load_obs_normalizer(vecnormalize_path, stage_index)

    ore_ids = [int(o) for o in ORE_TYPES]
    grid = _build_preference_grid()
    results: dict = {}

    for pref_name, pref_vec in grid:
        episode_rewards: list[float] = []
        episode_ores: list[int] = []
        episode_target_ores: list[int] = []
        episode_non_target_ores: list[int] = []
        episode_waste: list[int] = []
        episode_steps_list: list[int] = []
        episode_potentials: list[float] = []
        episode_completion: list[float] = []
        episode_world_type: str = "sim"

        # Auto-detect required biome for one-hot preferences
        required_biome = None
        if cache_dir is not None and np.max(pref_vec) == 1.0:
            target_idx = int(np.argmax(pref_vec))
            forced = ORE_TYPE_CONFIGS[target_idx].forced_biome
            if forced is not None:
                required_biome = int(forced)

        for ep in range(n_episodes):
            env_kwargs: dict = {
                "curriculum_stage": eval_stage,
                "seed": ep * 1000,
            }
            if world_class is not None:
                env_kwargs["world_class"] = world_class
                env_kwargs["cache_dir"] = cache_dir
                env_kwargs["required_biome"] = required_biome
            env = MinecraftMiningEnv(**env_kwargs)

            obs, info = env.reset()
            # Set preference on both obs and internal env state
            env._preference = pref_vec.copy()
            obs["pref"] = pref_vec.copy()
            if normalize is not None:
                obs = normalize(obs)

            total_reward = 0.0
            ore_count = 0
            target_ores = 0
            non_target_ores = 0
            waste = 0
            done = False
            ep_steps = 0
            final_potential = 0.0

            while not done:
                action_mask = env.action_masks()
                action, _ = model.predict(
                    obs,
                    action_masks=action_mask,
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = env.step(
                    int(action),
                )
                obs["pref"] = pref_vec.copy()
                if normalize is not None:
                    obs = normalize(obs)
                total_reward += reward

                if info.get("block_mined") is not None:
                    bt = info["block_mined"]
                    if bt in ore_ids:
                        ore_count += 1
                        idx = ore_ids.index(bt)
                        if pref_vec[idx] > 0:
                            target_ores += 1
                        else:
                            non_target_ores += 1
                    else:
                        waste += 1

                final_potential = info.get(
                    "harvest_potential", 0.0,
                )
                done = terminated or truncated
                ep_steps += 1

            episode_world_type = info.get("world_type", "sim")
            completion = info.get("completion_ratio", 0.0)

            episode_rewards.append(total_reward)
            episode_ores.append(ore_count)
            episode_target_ores.append(target_ores)
            episode_non_target_ores.append(non_target_ores)
            episode_waste.append(waste)
            episode_steps_list.append(ep_steps)
            episode_potentials.append(final_potential)
            episode_completion.append(completion)

        results[pref_name] = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_ores": float(np.mean(episode_ores)),
            "mean_target_ores": float(np.mean(episode_target_ores)),
            "mean_non_target_ores": float(np.mean(episode_non_target_ores)),
            "mean_waste": float(np.mean(episode_waste)),
            "mean_steps": float(np.mean(episode_steps_list)),
            "mean_potential": float(np.mean(episode_potentials)),
            "mean_completion": float(np.mean(episode_completion)),
            "world_type": episode_world_type,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mining agent")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model.zip",
    )
    parser.add_argument(
        "--vecnormalize", type=str, default=None,
        help="Path to vecnormalize.pkl",
    )
    parser.add_argument(
        "--stage", type=int, default=0,
        help="Curriculum stage for evaluation (training stage for VecNormalize)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Episodes per preference vector",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Path to chunk cache dir (enables real chunk evaluation)",
    )
    parser.add_argument(
        "--min-ores", type=int, default=1,
        help="Min ore count to accept a chunk (0 = no filter)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    results = evaluate(
        model_path=args.model,
        vecnormalize_path=args.vecnormalize,
        stage_index=args.stage,
        n_episodes=args.episodes,
        cache_dir=args.cache_dir,
        min_ores=args.min_ores,
    )

    # Determine world type for display
    any_result = next(iter(results.values()), {})
    world_type = any_result.get("world_type", "sim")
    print(f"\nWorld type: {world_type}")

    header = (
        f"{'Preference':<25} {'Reward':>10} {'Target':>8}"
        f" {'NonTgt':>8} {'Waste':>8} {'Steps':>8}"
        f" {'Complt%':>8}"
    )
    print(f"\n{header}")
    print("-" * 80)
    for name, metrics in results.items():
        print(
            f"{name:<25} "
            f"{metrics['mean_reward']:>10.2f} "
            f"{metrics['mean_target_ores']:>8.1f} "
            f"{metrics['mean_non_target_ores']:>8.1f} "
            f"{metrics['mean_waste']:>8.1f} "
            f"{metrics['mean_steps']:>8.1f} "
            f"{metrics['mean_completion'] * 100:>7.1f}%"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
