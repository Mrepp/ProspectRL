"""Multi-agent training entry point.

4-phase training progression:
  Phase 1: Single-agent task mastery (~4M steps)
  Phase 2: Multi-agent execution with heuristic coordinator (~2M steps)
  Phase 3: Coordinator GNN training with frozen agent (~2M steps)
  Phase 4: Joint training with blended assignment (~3M steps)

Usage::

    python train_multiagent.py --phase 1 --total-timesteps 4000000
    python train_multiagent.py --phase 1 --total-timesteps 5000 --n-envs 2 --dummy-vec-env
    python train_multiagent.py --phase 2 --n-agents 32 --total-timesteps 2000000
    python train_multiagent.py --phase 3 --n-agents 32 --total-timesteps 2000000
    python train_multiagent.py --phase 4 --n-agents 32 --total-timesteps 3000000
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from pathlib import Path

import numpy as np

logger = logging.getLogger("prospect_rl.train_multiagent")


def _configure_torch_backends() -> None:
    """Enable GPU-friendly defaults for Ada Lovelace / Ampere+ GPUs."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProspectRL Multi-Agent Training")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--n-agents", type=int, default=8)
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel envs for Phase 1 (default: auto)")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir")
    parser.add_argument("--freeze-agent", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_multiagent")
    parser.add_argument("--log-dir", type=str, default="./tb_logs_multiagent")
    parser.add_argument("--world-size", type=int, nargs=3, default=[64, 128, 64])
    parser.add_argument("--preference", type=float, nargs=8, default=None)
    parser.add_argument("--coordinator-k", type=int, default=50)
    parser.add_argument("--mining-radius", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device: auto, cuda, cpu")
    parser.add_argument("--dummy-vec-env", action="store_true",
                        help="Use DummyVecEnv instead of SubprocVecEnv")
    return parser.parse_args()


def train_phase1(args: argparse.Namespace) -> None:
    """Phase 1: Single-agent task mastery with MaskablePPO.

    Trains the agent policy to reliably complete individual tasks
    (MOVE_TO, EXCAVATE, MINE_ORE, RETURN_TO) in isolation using
    the Phase1TaskEnv wrapper and SB3 MaskablePPO.
    """
    _configure_torch_backends()

    from prospect_rl.models.callbacks import DriveCheckpointCallback
    from prospect_rl.models.ppo_config import linear_schedule
    from prospect_rl.multiagent.agent.agent_policy import MultiAgentFeatureExtractor
    from prospect_rl.multiagent.training.phase1_env import (
        Phase1DifficultyCallback,
        Phase1MetricsCallback,
        make_phase1_env,
    )
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import VecNormalize

    logger.info("Phase 1: Single-agent task mastery (MaskablePPO)")

    # Auto-scale n_envs
    if args.n_envs is not None:
        n_envs = args.n_envs
    else:
        cpu_count = multiprocessing.cpu_count()
        n_envs = max(8, min(48, (cpu_count * 2 // 8) * 8))

    world_size = tuple(args.world_size)
    total_timesteps = args.total_timesteps or 4_000_000
    use_subproc = not args.dummy_vec_env

    # Create environment
    env = make_phase1_env(
        n_envs=n_envs,
        world_size=world_size,
        seed=args.seed,
        max_episode_steps=1000,
        use_subproc=use_subproc,
    )

    # Resume or create new model
    reset_num_timesteps = True
    initial_difficulty = 0

    if args.resume:
        ckpt_dir = Path(args.resume)
        latest_path = ckpt_dir / "latest.json"
        if latest_path.exists():
            manifest = json.loads(latest_path.read_text())
            logger.info("Resuming from step %d", manifest["step"])

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

            # Estimate difficulty from saved timestep
            progress = manifest["step"] / total_timesteps
            if progress >= 0.66:
                initial_difficulty = 2
            elif progress >= 0.33:
                initial_difficulty = 1
        else:
            logger.warning("No checkpoint found at %s, starting fresh", ckpt_dir)
            model = _create_phase1_model(env, args, total_timesteps)
    else:
        model = _create_phase1_model(env, args, total_timesteps)

    # Callbacks
    ckpt_dir = Path(args.output_dir) / "phase1"
    callbacks = CallbackList([
        DriveCheckpointCallback(
            checkpoint_dir=str(ckpt_dir),
            save_freq=25_000,
            max_kept=3,
            verbose=1,
        ),
        Phase1DifficultyCallback(initial_difficulty=initial_difficulty, verbose=1),
        Phase1MetricsCallback(verbose=0),
    ])

    # Train
    vec_type = "SubprocVecEnv" if use_subproc else "DummyVecEnv"
    logger.info(
        "Training: envs=%d (%s), device=%s, timesteps=%d, seed=%d",
        n_envs, vec_type, args.device, total_timesteps, args.seed,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )

    # Save final checkpoint
    final_dir = ckpt_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(final_dir / "model.zip"))
    env.save(str(final_dir / "vecnormalize.pkl"))
    logger.info("Phase 1 complete. Final model saved to %s", final_dir)


def _create_phase1_model(
    env: object,
    args: argparse.Namespace,
    total_timesteps: int,
) -> "MaskablePPO":
    """Instantiate a fresh MaskablePPO for Phase 1."""
    from prospect_rl.models.ppo_config import linear_schedule
    from prospect_rl.multiagent.agent.agent_policy import MultiAgentFeatureExtractor
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=linear_schedule(3e-4),
        n_steps=1024,
        batch_size=2048,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.10,
        vf_coef=0.75,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs={
            "features_extractor_class": MultiAgentFeatureExtractor,
            "net_arch": {"pi": [64, 64], "vf": [128, 128]},
        },
        seed=args.seed,
        tensorboard_log=args.log_dir,
        device=args.device,
        verbose=1,
    )


def _get_preference(args: argparse.Namespace) -> np.ndarray:
    """Build ore preference vector from CLI args."""
    preference = np.zeros(8, dtype=np.float32)
    if args.preference:
        preference = np.array(args.preference, dtype=np.float32)
    else:
        preference[3] = 1.0  # Diamond
    return preference


def _create_phase2_vecenv(
    args: argparse.Namespace,
    n_agents: int,
    preference: np.ndarray,
) -> "MultiAgentVecEnv":
    """Create a MultiAgentVecEnv for Phase 2+ training."""
    from prospect_rl.env.world.world import World
    from prospect_rl.multiagent.config import MultiAgentConfig
    from prospect_rl.multiagent.training.multi_agent_vec_env import MultiAgentVecEnv

    config = MultiAgentConfig()
    world_size = tuple(args.world_size)
    world = World(size=world_size, seed=args.seed, ore_density_multiplier=3.0)

    task_reward_config = {
        "completion_reward": config.task_completion_reward,
        "progress_reward": config.task_progress_reward,
        "block_cleared_reward": config.block_cleared_reward,
        "regress_penalty": config.regress_penalty,
        "idle_penalty": config.idle_penalty,
        "box_stay_bonus": config.box_stay_bonus,
        "box_leave_penalty": config.box_leave_penalty,
    }

    return MultiAgentVecEnv(
        world=world,
        n_agents=n_agents,
        preference=preference,
        coordinator_interval_k=args.coordinator_k,
        mining_radius=args.mining_radius,
        seed=args.seed,
        task_reward_config=task_reward_config,
        congestion_radius=config.astar_congestion_radius,
        excavate_ore_threshold=config.excavate_ore_threshold,
    )


def _load_phase1_model(args: argparse.Namespace) -> object:
    """Load the Phase 1 trained model for inference."""
    from sb3_contrib import MaskablePPO

    phase1_dir = Path(args.output_dir) / "phase1" / "final"
    model_path = phase1_dir / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Phase 1 model not found at {model_path}. Run Phase 1 first."
        )

    model = MaskablePPO.load(str(model_path), device=args.device)
    logger.info("Loaded Phase 1 model from %s", model_path)
    return model


def train_phase2(args: argparse.Namespace) -> None:
    """Phase 2: Multi-agent execution with heuristic coordinator.

    Loads the Phase 1 trained policy and uses it for agent inference.
    Agent count curriculum: 4 -> 8 -> 16 -> 32 every 500K steps.
    """
    import torch
    from prospect_rl.multiagent.config import MultiAgentConfig

    config = MultiAgentConfig()
    preference = _get_preference(args)
    total = args.total_timesteps or config.phase2_timesteps

    # Load trained Phase 1 model
    model = _load_phase1_model(args)

    # Explicitly freeze policy — Phase 2 is inference only
    for param in model.policy.parameters():
        param.requires_grad_(False)

    # Agent count curriculum
    schedule = config.agent_count_schedule
    interval = config.agent_count_step_interval
    schedule_idx = 0
    current_n = schedule[0]

    logger.info(
        "Phase 2: Multi-agent execution (start=%d agents, total=%d steps)",
        current_n, total,
    )

    vec_env = _create_phase2_vecenv(args, current_n, preference)
    obs_dict, _info = vec_env.reset()
    step = 0
    last_logged = 0
    last_ckpt = 0
    ckpt_dir = Path(args.output_dir) / "phase2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    while step < total:
        # Agent count scaling
        new_idx = min(len(schedule) - 1, step // interval)
        if new_idx > schedule_idx:
            schedule_idx = new_idx
            current_n = schedule[schedule_idx]
            logger.info("Scaling to %d agents at step %d", current_n, step)
            vec_env.close()
            vec_env = _create_phase2_vecenv(args, current_n, preference)
            obs_dict, _info = vec_env.reset()

        # Action selection via trained model
        masks = vec_env.get_action_masks()
        actions, _ = model.predict(obs_dict, action_masks=masks, deterministic=False)
        obs_dict, rewards, terms, truncs, infos = vec_env.step(actions)
        step += vec_env.num_envs

        # Logging
        if step - last_logged >= 10_000:
            last_logged = step
            avg_reward = float(np.mean(rewards))
            logger.info(
                "Step %d | agents=%d | Avg step reward: %.4f",
                step, current_n, avg_reward,
            )

        # Checkpointing
        if step - last_ckpt >= 100_000:
            last_ckpt = step
            ckpt_path = ckpt_dir / f"step_{step}"
            ckpt_path.mkdir(exist_ok=True)
            torch.save(
                vec_env._coordinator._gnn.state_dict(),
                str(ckpt_path / "coordinator.pt"),
            )
            manifest = {"step": step, "n_agents": current_n, "phase": 2}
            (ckpt_path / "manifest.json").write_text(json.dumps(manifest))
            logger.info("Checkpoint saved at step %d", step)

    vec_env.close()
    logger.info("Phase 2 complete at step %d", step)


def train_phase3(args: argparse.Namespace) -> None:
    """Phase 3: Coordinator GNN training (agents frozen).

    Freezes the agent policy and trains the GNN coordinator via
    REINFORCE on team-level reward.
    """
    import torch
    from prospect_rl.multiagent.config import MultiAgentConfig
    from prospect_rl.multiagent.training.coordinator_trainer import CoordinatorTrainer

    config = MultiAgentConfig()
    preference = _get_preference(args)
    total = args.total_timesteps or config.phase3_timesteps
    n_agents = args.n_agents or 32

    logger.info(
        "Phase 3: Coordinator GNN training (agents frozen, %d agents, %d steps)",
        n_agents, total,
    )

    # Load and freeze agent policy — Phase 3 only trains the GNN
    model = _load_phase1_model(args)
    for param in model.policy.parameters():
        param.requires_grad_(False)

    # Create env with 32 agents
    vec_env = _create_phase2_vecenv(args, n_agents, preference)

    # Create REINFORCE trainer for GNN
    trainer = CoordinatorTrainer(
        gnn=vec_env._coordinator._gnn,
        lr=config.coordinator_lr,
        ent_coef=config.coordinator_ent_coef,
    )

    obs_dict, _info = vec_env.reset()
    step = 0
    last_logged = 0
    last_ckpt = 0
    window_reward = 0.0
    ckpt_dir = Path(args.output_dir) / "phase3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    while step < total:
        masks = vec_env.get_action_masks()
        actions, _ = model.predict(obs_dict, action_masks=masks, deterministic=False)
        obs_dict, rewards, terms, truncs, infos = vec_env.step(actions)
        window_reward += float(np.sum(rewards))
        step += vec_env.num_envs

        # Train GNN on coordinator replan
        coordinator = vec_env._coordinator
        if coordinator.step_since_replan == 0 and coordinator._last_x_dict is not None:
            metrics = trainer.update(
                x_dict=coordinator._last_x_dict,
                edge_index_dict=coordinator._last_edge_index_dict,
                row_idx=coordinator._last_row_idx,
                col_idx=coordinator._last_col_idx,
                team_reward=window_reward,
            )
            window_reward = 0.0

        # Logging
        if step - last_logged >= 10_000:
            last_logged = step
            avg_reward = float(np.mean(rewards))
            log_msg = f"Step {step} | Avg reward: {avg_reward:.4f}"
            if trainer.total_updates > 0:
                log_msg += (
                    f" | GNN loss: {trainer._last_loss:.4f}"
                    f" | entropy: {trainer._last_entropy:.4f}"
                )
            logger.info(log_msg)

        # Checkpointing
        if step - last_ckpt >= 100_000:
            last_ckpt = step
            ckpt_path = ckpt_dir / f"step_{step}"
            ckpt_path.mkdir(exist_ok=True)
            torch.save(
                coordinator._gnn.state_dict(),
                str(ckpt_path / "coordinator.pt"),
            )
            manifest = {
                "step": step, "n_agents": n_agents, "phase": 3,
                "gnn_updates": trainer.total_updates,
            }
            (ckpt_path / "manifest.json").write_text(json.dumps(manifest))
            logger.info("Checkpoint saved at step %d", step)

    vec_env.close()
    logger.info(
        "Phase 3 complete at step %d (%d GNN updates)", step, trainer.total_updates,
    )


def train_phase4(args: argparse.Namespace) -> None:
    """Phase 4: Joint training with blended assignment.

    Trains both agent policy and coordinator jointly. Uses blended
    heuristic/GNN assignment with alpha schedule.
    """
    import torch
    from prospect_rl.multiagent.config import MultiAgentConfig
    from prospect_rl.multiagent.training.coordinator_trainer import CoordinatorTrainer

    config = MultiAgentConfig()
    preference = _get_preference(args)
    total = args.total_timesteps or config.phase4_timesteps
    n_agents = args.n_agents or 32
    alpha_schedule = config.blend_alpha_schedule
    alpha_interval = config.blend_alpha_step_interval

    logger.info(
        "Phase 4: Joint training (%d agents, %d steps, alpha=%s)",
        n_agents, total, alpha_schedule,
    )

    # Load agent model (now trainable) and coordinator
    model = _load_phase1_model(args)

    # Try to load Phase 3 coordinator checkpoint
    phase3_dir = Path(args.output_dir) / "phase3"
    phase3_ckpts = sorted(phase3_dir.glob("step_*/coordinator.pt"))
    vec_env = _create_phase2_vecenv(args, n_agents, preference)

    if phase3_ckpts:
        latest = phase3_ckpts[-1]
        vec_env._coordinator._gnn.load_state_dict(torch.load(str(latest)))
        logger.info("Loaded Phase 3 coordinator from %s", latest)

    trainer = CoordinatorTrainer(
        gnn=vec_env._coordinator._gnn,
        lr=config.coordinator_lr,
        ent_coef=config.coordinator_ent_coef,
    )

    # Set initial alpha for blended assignment
    vec_env._coordinator.alpha = alpha_schedule[0]

    obs_dict, _info = vec_env.reset()
    step = 0
    last_logged = 0
    last_ckpt = 0
    window_reward = 0.0
    alpha_idx = 0
    ckpt_dir = Path(args.output_dir) / "phase4"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    while step < total:
        # Update alpha schedule
        new_idx = min(len(alpha_schedule) - 1, step // alpha_interval)
        if new_idx > alpha_idx:
            alpha_idx = new_idx
            vec_env._coordinator.alpha = alpha_schedule[alpha_idx]
            logger.info(
                "Blend alpha -> %.2f at step %d",
                alpha_schedule[alpha_idx], step,
            )

        masks = vec_env.get_action_masks()
        actions, _ = model.predict(obs_dict, action_masks=masks, deterministic=False)
        obs_dict, rewards, terms, truncs, infos = vec_env.step(actions)
        window_reward += float(np.sum(rewards))
        step += vec_env.num_envs

        # Train GNN on coordinator replan
        coordinator = vec_env._coordinator
        if coordinator.step_since_replan == 0 and coordinator._last_x_dict is not None:
            trainer.update(
                x_dict=coordinator._last_x_dict,
                edge_index_dict=coordinator._last_edge_index_dict,
                row_idx=coordinator._last_row_idx,
                col_idx=coordinator._last_col_idx,
                team_reward=window_reward,
            )
            window_reward = 0.0

        # Logging
        if step - last_logged >= 10_000:
            last_logged = step
            avg_reward = float(np.mean(rewards))
            logger.info(
                "Step %d | alpha=%.2f | Avg reward: %.4f",
                step, alpha_schedule[alpha_idx], avg_reward,
            )

        # Checkpointing
        if step - last_ckpt >= 100_000:
            last_ckpt = step
            ckpt_path = ckpt_dir / f"step_{step}"
            ckpt_path.mkdir(exist_ok=True)
            torch.save(
                coordinator._gnn.state_dict(),
                str(ckpt_path / "coordinator.pt"),
            )
            manifest = {
                "step": step, "n_agents": n_agents, "phase": 4,
                "alpha": alpha_schedule[alpha_idx],
            }
            (ckpt_path / "manifest.json").write_text(json.dumps(manifest))
            logger.info("Checkpoint saved at step %d", step)

    vec_env.close()
    logger.info("Phase 4 complete at step %d", step)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    phase_fn = {1: train_phase1, 2: train_phase2, 3: train_phase3, 4: train_phase4}
    phase_fn[args.phase](args)


if __name__ == "__main__":
    main()
