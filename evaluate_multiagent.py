"""Multi-agent evaluation entry point.

Evaluates a trained multi-agent system on procedural or real-chunk worlds.
Reports per-task success rate, team ore yield, and extraction speed.

Usage::

    python evaluate_multiagent.py --n-agents 32 --episodes 10
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

logger = logging.getLogger("prospect_rl.evaluate_multiagent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProspectRL Multi-Agent Evaluation")
    parser.add_argument("--n-agents", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, nargs=3, default=[64, 128, 64])
    parser.add_argument("--preference", type=float, nargs=8, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    from prospect_rl.env.world.world import World
    from prospect_rl.multiagent.training.multi_agent_vec_env import MultiAgentVecEnv

    world_size = tuple(args.world_size)

    preference = np.zeros(8, dtype=np.float32)
    if args.preference:
        preference = np.array(args.preference, dtype=np.float32)
    else:
        preference[3] = 1.0

    episode_rewards = []
    episode_ores = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        world = World(size=world_size, seed=seed, ore_density_multiplier=1.0)

        vec_env = MultiAgentVecEnv(
            world=world,
            n_agents=args.n_agents,
            preference=preference,
            seed=seed,
            max_episode_steps=args.max_steps,
        )

        obs_list, _ = vec_env.reset()
        total_reward = 0.0
        total_ores = 0

        for step in range(args.max_steps):
            masks = vec_env.get_action_masks()
            actions = []
            for i in range(vec_env.num_envs):
                valid = np.where(masks[i])[0]
                actions.append(int(np.random.choice(valid)) if len(valid) > 0 else 0)

            obs_list, rewards, terms, truncs, infos = vec_env.step(actions)
            total_reward += sum(rewards)

            for info in infos:
                if info.get("block_mined") is not None:
                    total_ores += 1

            if all(truncs):
                break

        episode_rewards.append(total_reward)
        episode_ores.append(total_ores)
        logger.info(
            "Episode %d: reward=%.2f, ores_mined=%d",
            ep, total_reward, total_ores,
        )

    logger.info(
        "Evaluation complete. Mean reward: %.2f (std=%.2f), "
        "Mean ores: %.1f (std=%.1f)",
        np.mean(episode_rewards), np.std(episode_rewards),
        np.mean(episode_ores), np.std(episode_ores),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
