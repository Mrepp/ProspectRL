"""Multi-agent system configuration.

Centralises all hyperparameters for the Bayesian coordinator,
GNN policy, A* pathfinding, belief map, and agent extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MultiAgentConfig:
    """Top-level configuration for the multi-agent mining system."""

    # --- Coordinator scheduling ---
    coordinator_interval_k: int = 50
    replan_on_event: bool = True

    # --- GNN Coordinator ---
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_heads: int = 4
    coordinator_lr: float = 3e-4
    coordinator_ent_coef: float = 0.05

    # --- Bounding box constraints ---
    region_constraints: list[dict] | None = None

    # --- Belief map ---
    chunk_size_xz: int = 16
    cluster_strength: float = 0.3
    cluster_radius: float = 4.0
    depletion_epsilon: float = 0.1
    belief_invalidation_radius: int = 1

    # --- A* pathfinding ---
    astar_max_iterations: int = 5000
    astar_dig_cost: float = 3.0
    astar_congestion_cost: float = 2.0
    astar_congestion_radius: int = 3
    mining_radius: int = 8

    # --- Agent observation extensions ---
    # rel_xyz(3) + distance(1) + task_onehot(4) + boundary_dist(1) + inside_flag(1)
    nav_target_extra_dims: int = 10

    # --- Task rewards ---
    task_completion_reward: float = 1.0
    task_progress_reward: float = 0.1
    block_cleared_reward: float = 0.05
    regress_penalty: float = -0.02
    idle_penalty: float = -0.01
    box_stay_bonus: float = 0.05
    box_leave_penalty: float = -0.2
    excavate_ore_threshold: float = 1.0

    # --- Distribution-aware MINE_ORE rewards ---
    alignment_bonus: float = 0.05
    rarity_mult_cap: float = 5.0
    alignment_min_ores: int = 3
    preference_blend_alpha: float = 0.5

    # --- Task sampling (Phase 1 training) ---
    task_sample_weights: dict[str, float] = field(default_factory=lambda: {
        "MOVE_TO": 0.35,
        "EXCAVATE": 0.40,
        "MINE_ORE": 0.20,
        "RETURN_TO": 0.05,
    })

    # --- Scale ---
    max_agents: int = 100
    min_spawn_distance: int = 5
    heartbeat_timeout: float = 30.0

    # --- Training phases ---
    phase1_timesteps: int = 4_000_000
    phase2_timesteps: int = 2_000_000
    phase3_timesteps: int = 2_000_000
    phase4_timesteps: int = 3_000_000

    # --- Agent count curriculum (Phase 2+) ---
    agent_count_schedule: list[int] = field(
        default_factory=lambda: [4, 8, 16, 32],
    )
    agent_count_step_interval: int = 500_000

    # --- Phase 4 blended assignment ---
    blend_alpha_schedule: list[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.6, 1.0],
    )
    blend_alpha_step_interval: int = 750_000
