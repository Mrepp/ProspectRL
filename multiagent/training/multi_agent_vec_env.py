"""Multi-agent vectorized environment.

Single-process VecEnv where all agents share one world.
Batches RL-mode agents for a single GPU forward pass while
A*-mode agents are computed on CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from prospect_rl.config import NUM_ACTIONS, NUM_ORE_TYPES, BlockType
from prospect_rl.env.turtle import Turtle
from prospect_rl.multiagent.agent.multi_agent_env import MultiAgentMiningEnv
from prospect_rl.multiagent.belief_map import BeliefMap
from prospect_rl.multiagent.coordinator.coordinator import Coordinator
from prospect_rl.multiagent.geological_prior import AnalyticalPrior
from prospect_rl.multiagent.shared_world import SharedWorld


class MultiAgentVecEnv:
    """Vectorized environment for multi-agent mining.

    All agents operate in the same ``SharedWorld``. The coordinator
    runs every ``K`` steps to reassign tasks. A*-mode agents get
    their action from the pathfinder; RL-mode agents are batched
    for a single GPU inference.

    Parameters
    ----------
    world:
        The underlying world object.
    n_agents:
        Number of agents.
    preference:
        Global 8-dim ore preference.
    coordinator_interval_k:
        Steps between coordinator replans.
    mining_radius:
        Distance threshold for A*/RL mode switch.
    max_episode_steps:
        Max steps per episode.
    seed:
        Random seed.
    """

    def __init__(
        self,
        world: object,
        n_agents: int = 8,
        preference: np.ndarray | None = None,
        coordinator_interval_k: int = 50,
        mining_radius: int = 8,
        max_episode_steps: int = 1000,
        seed: int = 42,
        task_reward_config: dict[str, float] | None = None,
        congestion_radius: int = 3,
        excavate_ore_threshold: float = 1.0,
    ) -> None:
        self._n_agents = n_agents
        self._coordinator_k = coordinator_interval_k
        self._global_step = 0

        # Shared world
        self._shared_world = SharedWorld(world, max_agents=n_agents + 10)

        # Preference
        if preference is None:
            preference = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
            preference[3] = 1.0  # Default: diamond
        self._preference = preference

        # Belief map + prior
        sx, sy, sz = self._shared_world.shape
        self._prior = AnalyticalPrior(world_height=sy)
        self._belief_map = BeliefMap(
            world_size=(sx, sy, sz),
            biome_map=self._shared_world.biome_map,
            prior=self._prior,
        )

        # Coordinator
        self._coordinator = Coordinator(
            belief_map=self._belief_map,
            shared_world=self._shared_world,
            preference=self._preference,
            congestion_radius=congestion_radius,
            excavate_ore_threshold=excavate_ore_threshold,
        )
        self._task_reward_config = task_reward_config or {}

        # Spawn agents
        rng = np.random.default_rng(seed)
        spawn_positions = self._shared_world.spawn_positions(
            n_agents, min_distance=5, rng=rng,
        )

        # Create per-agent environments
        self._envs: list[MultiAgentMiningEnv] = []
        for i in range(n_agents):
            pos = spawn_positions[i] if i < len(spawn_positions) else np.array(
                [sx // 2, sy // 2, sz // 2], dtype=np.int32,
            )
            turtle = Turtle(
                position=pos, facing=0,
                fuel=10000, max_fuel=10000,
            )
            self._shared_world.register_agent(i, turtle)

            env = MultiAgentMiningEnv(
                agent_id=i,
                shared_world=self._shared_world,
                mining_radius=mining_radius,
                max_episode_steps=max_episode_steps,
                task_reward_config=self._task_reward_config,
            )
            self._envs.append(env)

        # Expose observation/action spaces for SB3 compatibility
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space

    @property
    def num_envs(self) -> int:
        return self._n_agents

    def _stack_obs(self, obs_list: list[dict]) -> dict[str, np.ndarray]:
        """Stack per-agent observations into batched arrays."""
        return {key: np.stack([o[key] for o in obs_list]) for key in obs_list[0]}

    def reset(self) -> tuple[dict[str, np.ndarray], list[dict]]:
        """Reset all agents. Returns (stacked observations, infos)."""
        obs_list = []
        info_list = []

        for env in self._envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        # Initial coordinator plan
        self._coordinator.plan(step=0)
        for env in self._envs:
            assignment = self._coordinator.get_assignment(env._agent_id)
            if assignment is not None:
                env.set_assignment(assignment)

        self._global_step = 0
        return self._stack_obs(obs_list), info_list

    def step(
        self, actions: np.ndarray | list[int],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all agents with given actions.

        For A*-mode agents, the provided action is ignored and replaced
        with the pathfinder action. Callers should check ``is_astar_mode``
        to decide whether to query the RL policy.

        Returns
        -------
        observations : dict[str, np.ndarray]
            Stacked observations keyed by "voxels", "scalars", "pref".
        rewards : np.ndarray
            Shape (n_agents,).
        terminated : np.ndarray
            Shape (n_agents,).
        truncated : np.ndarray
            Shape (n_agents,).
        infos : list[dict]
            Per-agent info dicts.
        """
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []

        for i, env in enumerate(self._envs):
            # Override action with A* if in navigation mode.
            # NOTE: Phase 1 has no A* mode (single-agent tasks).
            # Phases 2-4 use manual loops without model.learn(),
            # so A*-selected actions don't pollute PPO training.
            # If Phase 4 adds model.learn(), A* transitions
            # (info["was_astar_action"]) should be masked out.
            action = int(actions[i])
            was_astar = env.is_astar_mode
            if was_astar:
                astar_action = env.get_astar_action()
                if astar_action is not None:
                    action = astar_action

            obs, reward, terminated, truncated, info = env.step(action)
            info["was_astar_action"] = was_astar

            # Auto-reset terminated/truncated agents
            if terminated or truncated:
                info["terminal_observation"] = obs
                # Clear old occupancy before reset to prevent ghost entries
                self._shared_world.deregister_agent(env._agent_id)
                obs, _reset_info = env.reset()
                # Re-assign task from coordinator
                assignment = self._coordinator.get_assignment(env._agent_id)
                if assignment is not None:
                    env.set_assignment(assignment)

            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)

        # Process telemetry and update belief map
        all_events = self._shared_world.flush_telemetry()
        for agent_id, events in all_events.items():
            self._belief_map.process_events(events, step=self._global_step)

        # Coordinator replan check
        self._global_step += 1
        self._shared_world.increment_step()
        self._coordinator.increment_step()

        if self._coordinator.step_since_replan >= self._coordinator_k:
            self._coordinator.plan(step=self._global_step)
            for env in self._envs:
                assignment = self._coordinator.get_assignment(env._agent_id)
                if assignment is not None:
                    env.set_assignment(assignment)

        return (
            self._stack_obs(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(terminated_list, dtype=bool),
            np.array(truncated_list, dtype=bool),
            info_list,
        )

    def get_astar_mask(self) -> np.ndarray:
        """Return boolean array indicating which agents are in A* mode."""
        return np.array([env.is_astar_mode for env in self._envs])

    def get_action_masks(self) -> np.ndarray:
        """Return action masks for all agents, shape (n_agents, NUM_ACTIONS)."""
        masks = np.ones((self._n_agents, NUM_ACTIONS), dtype=bool)
        for i, env in enumerate(self._envs):
            masks[i] = env.action_masks()
        return masks

    def close(self) -> None:
        """Clean up resources."""
        pass

    def env_method(
        self, method_name: str, *args: Any, indices: list[int] | None = None, **kwargs: Any,
    ) -> list[Any]:
        """Call a method on each sub-environment."""
        targets = range(self._n_agents) if indices is None else indices
        return [getattr(self._envs[i], method_name)(*args, **kwargs) for i in targets]

    def get_attr(self, attr_name: str, indices: list[int] | None = None) -> list[Any]:
        """Get an attribute from each sub-environment."""
        targets = range(self._n_agents) if indices is None else indices
        return [getattr(self._envs[i], attr_name) for i in targets]

    def set_attr(
        self, attr_name: str, value: Any, indices: list[int] | None = None,
    ) -> None:
        """Set an attribute on each sub-environment."""
        targets = range(self._n_agents) if indices is None else indices
        for i in targets:
            setattr(self._envs[i], attr_name, value)
