"""Evaluate a trained model across a preference grid.

Usage::

    python -m prospect_rl.evaluate --model ./checkpoints/final/model.zip
    python -m prospect_rl.evaluate --model ./checkpoints/final/model.zip --episodes 20
"""

from __future__ import annotations

import argparse

import numpy as np
from prospect_rl.config import NUM_ORE_TYPES, ORE_TYPES


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


def evaluate(
    model_path: str,
    vecnormalize_path: str | None = None,
    stage_index: int = 0,
    n_episodes: int = 10,
) -> dict:
    """Run evaluation across the preference grid.

    Returns a dict mapping preference name to per-episode metrics.
    """
    from prospect_rl.env.mining_env import MinecraftMiningEnv
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path)

    grid = _build_preference_grid()
    results: dict = {}

    for pref_name, pref_vec in grid:
        episode_rewards = []
        episode_ores = []
        episode_steps_list = []

        for ep in range(n_episodes):
            env = MinecraftMiningEnv(
                curriculum_stage=stage_index,
                seed=ep * 1000,
            )

            obs, info = env.reset()
            # Override preference
            obs["pref"] = pref_vec.copy()

            total_reward = 0.0
            ore_count = 0
            done = False
            ep_steps = 0

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
                total_reward += reward
                if info.get("block_mined") is not None:
                    bt = info["block_mined"]
                    if bt in [int(o) for o in ORE_TYPES]:
                        ore_count += 1
                done = terminated or truncated
                ep_steps += 1

            episode_rewards.append(total_reward)
            episode_ores.append(ore_count)
            episode_steps_list.append(ep_steps)

        results[pref_name] = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_ores": float(np.mean(episode_ores)),
            "mean_steps": float(np.mean(episode_steps_list)),
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
        help="Curriculum stage for evaluation",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Episodes per preference vector",
    )
    args = parser.parse_args()

    results = evaluate(
        model_path=args.model,
        vecnormalize_path=args.vecnormalize,
        stage_index=args.stage,
        n_episodes=args.episodes,
    )

    print(f"\n{'Preference':<25} {'Reward':>10} {'Ores':>8} {'Steps':>8}")
    print("-" * 55)
    for name, metrics in results.items():
        print(
            f"{name:<25} "
            f"{metrics['mean_reward']:>10.2f} "
            f"{metrics['mean_ores']:>8.1f} "
            f"{metrics['mean_steps']:>8.1f}"
        )


if __name__ == "__main__":
    main()
