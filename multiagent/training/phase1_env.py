"""Phase 1 single-agent task mastery environment and factory.

Wraps ``MultiAgentMiningEnv`` so that each episode creates a fresh world,
auto-assigns tasks from a ``TaskSampler``, and re-assigns when tasks
complete or time-out.  Designed for SB3 ``MaskablePPO`` via SubprocVecEnv.
"""

from __future__ import annotations

import multiprocessing
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from prospect_rl.config import NUM_ORE_TYPES, BlockType
from prospect_rl.env.turtle import Turtle
from prospect_rl.env.world.world import World
from prospect_rl.multiagent.agent.multi_agent_env import MultiAgentMiningEnv
from prospect_rl.multiagent.coordinator.assignment import TaskType
from prospect_rl.multiagent.shared_world import SharedWorld
from prospect_rl.multiagent.training.task_sampler import TaskSampler


class Phase1TaskEnv(gym.Env):
    """Single-agent task-mastery environment for Phase 1.

    Creates a fresh ``World`` + ``SharedWorld`` each episode and auto-cycles
    through tasks sampled from a ``TaskSampler``.  A* is disabled
    (``mining_radius=999``) so the RL policy must learn all navigation.

    Parameters
    ----------
    world_size:
        (sx, sy, sz) world dimensions.
    ore_density_multiplier:
        Ore density multiplier for world generation.
    fixed_ore_index:
        If set, use a fixed one-hot preference for this ore index.
    max_episode_steps:
        Maximum total steps per episode.
    seed:
        Random seed.
    task_weights:
        Override default task-type sampling weights.
    task_step_limit:
        Max steps per individual task before forced re-sample.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world_size: tuple[int, int, int] = (64, 128, 64),
        ore_density_multiplier: float = 3.0,
        fixed_ore_index: int | None = None,
        max_episode_steps: int = 1000,
        seed: int = 42,
        task_weights: dict[str, float] | None = None,
        task_step_limit: int = 200,
    ) -> None:
        super().__init__()

        self._world_size = world_size
        self._ore_density_mult = ore_density_multiplier
        self._fixed_ore_index = fixed_ore_index
        self._max_episode_steps = max_episode_steps
        self._task_step_limit = task_step_limit

        self._rng = np.random.default_rng(seed)
        self._sampler = TaskSampler(
            world_size=world_size,
            weights=task_weights,
            seed=int(self._rng.integers(2**31)),
        )

        # Build a one-hot preference vector
        self._preference = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        if fixed_ore_index is not None:
            self._preference[fixed_ore_index % NUM_ORE_TYPES] = 1.0
        else:
            self._preference[0] = 1.0

        # Build initial world to define obs/action spaces
        self._shared_world: SharedWorld | None = None
        self._inner_env: MultiAgentMiningEnv | None = None
        self._build_env()

        # Copy spaces from inner env
        self.observation_space = self._inner_env.observation_space
        self.action_space = self._inner_env.action_space

        # Episode counters
        self._tasks_completed = 0
        self._current_task_steps = 0

    def _build_env(self) -> None:
        """Create a fresh World, SharedWorld, Turtle, and inner env."""
        world = World(
            size=self._world_size,
            seed=int(self._rng.integers(2**31)),
            ore_density_multiplier=self._ore_density_mult,
            caves_enabled=False,
        )
        self._shared_world = SharedWorld(world)

        # Spawn turtle near center with jitter
        sx, sy, sz = self._world_size
        cx, cy, cz = sx // 2, sy // 2, sz // 2
        jx = int(self._rng.integers(-sx // 4, sx // 4 + 1))
        jy = int(self._rng.integers(-sy // 4, sy // 4 + 1))
        jz = int(self._rng.integers(-sz // 4, sz // 4 + 1))
        px = max(1, min(sx - 2, cx + jx))
        py = max(1, min(sy - 2, cy + jy))
        pz = max(1, min(sz - 2, cz + jz))

        # Clear spawn position
        self._shared_world[px, py, pz] = BlockType.AIR

        turtle = Turtle(
            position=np.array([px, py, pz], dtype=np.int32),
            fuel=10000,
            max_fuel=10000,
        )
        self._shared_world.register_agent(0, turtle)

        self._inner_env = MultiAgentMiningEnv(
            agent_id=0,
            shared_world=self._shared_world,
            mining_radius=999,  # Disable A* — RL controls everything
            max_episode_steps=self._max_episode_steps,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Build fresh world each episode
        self._build_env()

        obs, info = self._inner_env.reset()

        # Sample first task
        pos = info["position"]
        task = self._sampler.sample(
            agent_id=0,
            agent_position=pos,
            preference=self._preference,
        )
        self._inner_env.set_assignment(task)

        self._tasks_completed = 0
        self._current_task_steps = 0

        info["tasks_completed"] = 0
        info["current_task_type"] = task.task_type.name
        return obs, info

    def step(
        self, action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._inner_env.step(action)
        self._current_task_steps += 1

        # Check for task completion or timeout
        task_complete = info.get("task_complete", False)
        if task_complete or self._current_task_steps >= self._task_step_limit:
            # Record the ending task before re-sampling
            assignment = self._inner_env._assignment
            info["task_ended"] = True
            info["ended_task_type"] = (
                assignment.task_type.name if assignment is not None else "NONE"
            )
            info["ended_task_complete"] = task_complete
            info["ended_task_steps"] = self._current_task_steps

            self._tasks_completed += 1
            # Sample new task
            pos = info["position"]
            task = self._sampler.sample(
                agent_id=0,
                agent_position=pos,
                preference=self._preference,
            )
            self._inner_env.set_assignment(task)
            self._current_task_steps = 0
            info["current_task_type"] = task.task_type.name
        else:
            info["task_ended"] = False
            assignment = self._inner_env._assignment
            if assignment is not None:
                info["current_task_type"] = assignment.task_type.name
            else:
                info["current_task_type"] = "NONE"

        info["tasks_completed"] = self._tasks_completed
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return boolean action mask (required by MaskablePPO)."""
        return self._inner_env.action_masks()

    def set_difficulty(self, level: int) -> None:
        """Forward difficulty setting to the task sampler."""
        self._sampler.set_difficulty(level)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_phase1_env(
    n_envs: int = 16,
    world_size: tuple[int, int, int] = (64, 128, 64),
    seed: int = 42,
    max_episode_steps: int = 1000,
    use_subproc: bool = True,
) -> "VecNormalize":
    """Create a vectorised, normalised Phase 1 training environment.

    Mirrors the pattern in ``ppo_config.py:make_training_env()``.
    Each sub-env gets a deterministic ore preference via ``i % NUM_ORE_TYPES``.

    Returns
    -------
    VecNormalize
        Wrapped vectorised environment.  Only the ``scalars`` key
        is normalised; ``voxels`` and ``pref`` are left untouched.
    """
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )

    env_fns = []
    for i in range(n_envs):
        ore_idx = i % NUM_ORE_TYPES

        # Capture loop variables
        def _make(idx=i, oi=ore_idx):
            def _init():
                return Phase1TaskEnv(
                    world_size=world_size,
                    ore_density_multiplier=3.0,
                    fixed_ore_index=oi,
                    max_episode_steps=max_episode_steps,
                    seed=seed + idx,
                )
            return _init

        env_fns.append(_make())

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


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

from stable_baselines3.common.callbacks import BaseCallback


class Phase1DifficultyCallback(BaseCallback):
    """Advance TaskSampler difficulty based on timestep progress.

    Difficulty 0 → 1 at 33% of training, 1 → 2 at 66%.
    """

    def __init__(
        self,
        initial_difficulty: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._current_difficulty = initial_difficulty

    def _on_training_start(self) -> None:
        # Apply initial difficulty on resume
        if self._current_difficulty > 0:
            self.training_env.env_method(
                "set_difficulty", self._current_difficulty,
            )

    def _on_step(self) -> bool:
        total = self.locals.get("total_timesteps", 0)
        if total <= 0:
            return True

        progress = self.num_timesteps / total
        if progress >= 0.66 and self._current_difficulty < 2:
            self._current_difficulty = 2
            self.training_env.env_method("set_difficulty", 2)
            if self.verbose:
                print(f"[Phase1] Difficulty -> 2 at step {self.num_timesteps}")
        elif progress >= 0.33 and self._current_difficulty < 1:
            self._current_difficulty = 1
            self.training_env.env_method("set_difficulty", 1)
            if self.verbose:
                print(f"[Phase1] Difficulty -> 1 at step {self.num_timesteps}")

        self.logger.record("phase1/difficulty", self._current_difficulty)
        return True


class Phase1MetricsCallback(BaseCallback):
    """Log Phase 1 task-mastery metrics to TensorBoard.

    Tracks per-task-type completion rate and mean steps, plus aggregate
    episode-level metrics.
    """

    _WINDOW = 100

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        # Per-task-type rolling windows
        self._task_outcomes: dict[str, deque[bool]] = {}
        self._task_steps: dict[str, deque[int]] = {}

    def _ensure_type(self, task_type: str) -> None:
        if task_type not in self._task_outcomes:
            self._task_outcomes[task_type] = deque(maxlen=self._WINDOW)
            self._task_steps[task_type] = deque(maxlen=self._WINDOW)

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            # --- Per-task-end tracking ---
            if info.get("task_ended", False):
                tt = info.get("ended_task_type", "UNKNOWN")
                self._ensure_type(tt)
                self._task_outcomes[tt].append(
                    info.get("ended_task_complete", False),
                )
                self._task_steps[tt].append(
                    info.get("ended_task_steps", 0),
                )

            # --- Episode end: log aggregate metrics ---
            done = self.locals.get("dones", [False] * (i + 1))[i]
            if done:
                tasks = info.get("tasks_completed", 0)
                self.logger.record("phase1/tasks_per_episode", tasks)
                self.logger.record(
                    "phase1/blocks_cleared",
                    info.get("blocks_cleared", 0),
                )
                self.logger.record(
                    "phase1/explored_count",
                    info.get("explored_count", 0),
                )
                self.logger.record(
                    "phase1/episode_steps",
                    info.get("step", 0),
                )

        # --- Per-task-type metrics ---
        all_completions = 0
        all_total = 0
        for tt, outcomes in self._task_outcomes.items():
            if len(outcomes) == 0:
                continue
            rate = sum(outcomes) / len(outcomes)
            self.logger.record(f"phase1/{tt}_completion_rate", rate)
            all_completions += sum(outcomes)
            all_total += len(outcomes)

            steps_dq = self._task_steps[tt]
            if len(steps_dq) > 0:
                self.logger.record(
                    f"phase1/{tt}_mean_steps",
                    sum(steps_dq) / len(steps_dq),
                )

        # Overall completion rate
        if all_total > 0:
            self.logger.record(
                "phase1/task_completion_rate",
                all_completions / all_total,
            )

        return True
