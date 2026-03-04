"""CLI training entrypoint for the PPO mining agent.

Usage::

    python -m prospect_rl.train --stage 0 --n-envs 4 --total-timesteps 1000000
    python -m prospect_rl.train --stage 0 --resume --checkpoint-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

from prospect_rl.config import Config


def _configure_torch_backends() -> None:
    """Enable GPU-friendly defaults for Ada Lovelace / Ampere+ GPUs."""
    import torch

    # TF32: 2x matmul/conv throughput on Ampere+ tensor cores, negligible
    # precision loss for RL training.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cuDNN benchmark: safe because voxel observation shape is fixed.
    torch.backends.cudnn.benchmark = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO mining agent")
    parser.add_argument(
        "--stage", type=int, default=0,
        help="Curriculum stage index (0-4)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help="Number of parallel environments (default: auto-scale by CPU cores)",
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
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Path to chunk cache dir (enables real chunk training)",
    )
    parser.add_argument(
        "--real-fraction", type=float, default=0.0,
        help="Fraction of envs using real chunks in mixed mode (0.0-1.0)",
    )
    parser.add_argument(
        "--real-cache-dir", type=str, default=None,
        help="Cache dir for real chunk envs in mixed mode",
    )
    parser.add_argument(
        "--real-stage", type=int, default=None,
        help="Curriculum stage for real chunk envs (default: same as --stage)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="PyTorch device: auto, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--dummy-vec-env", action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv (for debugging)",
    )
    args = parser.parse_args()

    _configure_torch_backends()

    config = Config()

    # Auto-scale n_envs by CPU cores: 2 envs per core, rounded to multiple of
    # NUM_ORE_TYPES (8) so every ore type gets equal representation.
    if args.n_envs is not None:
        n_envs = args.n_envs
    else:
        cpu_count = multiprocessing.cpu_count()
        n_envs = max(8, min(48, (cpu_count * 2 // 8) * 8))

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

    # Determine world class
    world_class = None
    if args.cache_dir:
        from prospect_rl.env.world.real_chunk_world import RealChunkWorld
        world_class = RealChunkWorld

    use_subproc = not args.dummy_vec_env

    # Create environment
    env = make_training_env(
        n_envs=n_envs,
        stage_index=args.stage,
        seed=seed,
        world_class=world_class,
        cache_dir=args.cache_dir,
        real_fraction=args.real_fraction,
        real_cache_dir=args.real_cache_dir,
        real_stage_index=args.real_stage,
        use_subproc=use_subproc,
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
                device=args.device,
            )
            reset_num_timesteps = False
        else:
            print("No checkpoint found, starting fresh")
            model = create_ppo_model(
                env, config, tensorboard_log=tb_log, seed=seed,
                device=args.device,
            )
    else:
        model = create_ppo_model(
            env, config, tensorboard_log=tb_log, seed=seed,
            device=args.device,
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
    vec_type = "SubprocVecEnv" if use_subproc else "DummyVecEnv"
    print(f"Training: stage={args.stage}, envs={n_envs} ({vec_type}), "
          f"device={args.device}, timesteps={total_timesteps}, seed={seed}")
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
