"""CLI training entrypoint for the PPO mining agent.

Usage::

    python -m prospect_rl.train --stage 0 --n-envs 4 --total-timesteps 1000000
    python -m prospect_rl.train --stage 0 --resume --checkpoint-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prospect_rl.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO mining agent")
    parser.add_argument(
        "--stage", type=int, default=0,
        help="Curriculum stage index (0-4)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help="Number of parallel environments (default: from config)",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Total training timesteps (default: from config)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--tb-log", type=str, default=None,
        help="TensorBoard log directory (default: from config)",
    )
    args = parser.parse_args()

    config = Config()
    n_envs = args.n_envs or config.training.n_envs
    total_timesteps = args.total_timesteps or config.training.total_timesteps
    seed = args.seed if args.seed is not None else config.training.seed
    tb_log = args.tb_log or config.training.tensorboard_log

    # Lazy imports to keep CLI fast
    from prospect_rl.models.callbacks import (
        CurriculumCallback,
        DriveCheckpointCallback,
        MetricsCallback,
    )
    from prospect_rl.models.ppo_config import (
        create_ppo_model,
        make_training_env,
    )
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import VecNormalize

    # Create environment
    env = make_training_env(
        n_envs=n_envs,
        stage_index=args.stage,
        seed=seed,
    )

    # Resume or create new model
    reset_num_timesteps = True
    if args.resume:
        latest_path = Path(args.checkpoint_dir) / "latest.json"
        if latest_path.exists():
            manifest = json.loads(latest_path.read_text())
            print(f"Resuming from step {manifest['step']}")

            # Load VecNormalize stats
            vn_path = manifest["vecnormalize_path"]
            env = VecNormalize.load(vn_path, env.venv)
            env.training = True
            env.norm_reward = True

            # Load model
            model = MaskablePPO.load(
                manifest["model_path"],
                env=env,
            )
            reset_num_timesteps = False
        else:
            print("No checkpoint found, starting fresh")
            model = create_ppo_model(
                env, config, tensorboard_log=tb_log, seed=seed,
            )
    else:
        model = create_ppo_model(
            env, config, tensorboard_log=tb_log, seed=seed,
        )

    # Callbacks
    callbacks = CallbackList([
        DriveCheckpointCallback(
            checkpoint_dir=args.checkpoint_dir,
            save_freq=config.training.checkpoint_freq,
            max_kept=config.training.max_checkpoints_kept,
            verbose=1,
        ),
        MetricsCallback(verbose=0),
        CurriculumCallback(stage_index=args.stage, verbose=0),
    ])

    # Train
    print(f"Training: stage={args.stage}, envs={n_envs}, "
          f"timesteps={total_timesteps}, seed={seed}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )

    # Save final model
    final_dir = Path(args.checkpoint_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(final_dir / "model.zip"))
    env.save(str(final_dir / "vecnormalize.pkl"))
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
