# ProspectRL — CLAUDE.md

## Project Overview

PPO-based RL for ComputerCraft turtle mining in Minecraft. Supports both **single-agent** training (curriculum stages 0-5) and **multi-agent** coordination (32-100+ turtles in a shared world). A centralized Bayesian coordinator orchestrates agents via GNN-based task assignment, while individual agents use a FiLM+SE policy for local mining.

## Quick Reference

- **Single-agent entry points**: `train.py` (training), `evaluate.py` (eval), `deployment/inference_server.py` (serve)
- **Multi-agent entry points**: `train_multiagent.py` (4-phase training), `evaluate_multiagent.py` (eval)
- **Config source of truth**: `config.py` — all hyperparameters, block defs, curriculum stages, reward settings. `OreTypeConfig` / `ORE_TYPE_CONFIGS` centralizes per-ore metadata (Y-range, vein size, biome).
- **Multi-agent config**: `multiagent/config.py` — `MultiAgentConfig` dataclass with coordinator, belief map, A*, and task reward settings.
- **Reward logic**: `env/reward_vector.py` (single-agent), `multiagent/agent/task_rewards.py` (multi-agent task rewards)
- **Tests**: `pytest` from repo root (`tests/` for single-agent, `tests/multiagent/` for multi-agent)

## Architecture Overview

```
Single-Agent Pipeline (Stages 0-5):
  train.py → MinecraftMiningEnv → MiningFeatureExtractor → MaskablePPO

Multi-Agent Pipeline:
  Coordinator (GNN + Hungarian, every K=50 steps or on event)
    ├─ BeliefMap — Bernoulli(p_v) per voxel, Bayesian updates from telemetry
    ├─ AnalyticalPrior — P(ore|y,biome) from ORE_SPAWN_CONFIGS
    ├─ CoordinatorGNN — heterogeneous graph (agent nodes + region nodes)
    │   ├─ GATv2Conv message passing → per-(agent, region) utility scores
    │   └─ Trained end-to-end via REINFORCE on team mining reward
    ├─ Hungarian Matching — optimal agent→region assignment from GNN scores
    ├─ Bounding Box Constraints — user-defined spatial regions filter candidates
    └─ Outputs: per-agent TaskAssignment (target_pos, ore_pref, task_type)
         │
         ▼
  Agent (per turtle, every step)
    ├─ Telemetry Events — block_observed/removed/added/changed, path_blocked
    ├─ A* Pathfinder — congestion-aware cost field, dynamic replanning
    ├─ Adapted RL Policy — FiLM+SE MaskablePPO for local mining
    ├─ Mode Switch — A* while traveling, RL when within mining_radius
    └─ Reports: telemetry events + 3-block observations → coordinator
```

Design philosophy: Classical planning (A*, Bayesian belief) handles navigation and world modeling. A GNN-based coordinator (trained end-to-end via RL) handles global task assignment. The existing FiLM+SE policy handles local mining behavior.

## Multi-Agent System

### File Structure

```
multiagent/
├── config.py                          # MultiAgentConfig dataclass
├── telemetry.py                       # TelemetryEvent, TelemetryEventType
├── belief_map.py                      # BeliefMap (Bernoulli voxel model), ChunkState/Status
├── geological_prior.py                # AnalyticalPrior — P(ore|y,biome)
├── pathfinding.py                     # AStarPathfinder with congestion + dig-through
├── shared_world.py                    # SharedWorld with occupancy grid + telemetry buffer
├── agent/
│   ├── multi_agent_env.py             # Per-agent env with A*/RL hybrid, telemetry
│   ├── agent_policy.py                # MultiAgentFeatureExtractor (16ch, 80 scalars)
│   ├── task_rewards.py                # Task-specific reward computation
│   └── communication.py               # AgentMessage, MessageBuffer
├── coordinator/
│   ├── coordinator.py                 # Graph construction, GNN inference, Hungarian matching
│   ├── gnn.py                         # CoordinatorGNN (HeteroConv + GATv2Conv)
│   └── assignment.py                  # TaskAssignment, TaskType, BoundingBox dataclasses
└── training/
    ├── multi_agent_vec_env.py         # Single-process VecEnv, batched RL inference
    ├── coordinator_trainer.py         # REINFORCE trainer for GNN, EMA baseline
    └── task_sampler.py                # Weighted task sampling for Phase 1
```

### Telemetry Event System (`multiagent/telemetry.py`)

Turtles have limited sensing: `turtle.inspect()` (front), `inspectUp()`, `inspectDown()` — 3 blocks per step. Plus dig-through reveals. All world knowledge flows through these observations.

| Event Type | Trigger | Effect on BeliefMap |
|---|---|---|
| `BLOCK_OBSERVED` | 3-block inspection each step | Set p_v=1.0 (ore) or p_v=0.0 (non-ore) |
| `BLOCK_REMOVED` | Agent mined a block | Mark as AIR, update chunk mined counts |
| `BLOCK_ADDED` | Block appeared (gravel fall, player) | Treat as new observation |
| `BLOCK_CHANGED` | Block type mismatch vs belief | Invalidate + neighbors, revert to prior |
| `PATH_BLOCKED` | A* step failed (expected air, found solid) | Invalidate path, trigger A* replan |

### Probabilistic Belief Map (`multiagent/belief_map.py`)

Bernoulli voxel model: each voxel `v` has `p_v[ore_type]` = probability of containing that ore. Sparse storage — unobserved voxels implicitly use the geological prior.

**Voxel states**: Unobserved (p=prior), Observed no-ore (p=0), Observed ore (p=1 + cluster propagation to neighbors).

**Cluster propagation**: On ore find, boost unobserved neighbors: `p_u[ore] += λ * exp(-dist/r)` where λ=`cluster_strength` (0.3) and r = ore-specific radius from `OreTypeConfig.typical_vein_size`.

**Chunk states**: `UNKNOWN` → `PARTIALLY_EXPLORED` → `EXPLORED` (>80% observed) → `EXHAUSTED` (E_remaining < ε for all ores).

**Expected remaining**: `E_remaining[ore] = Σ p_v[ore]` over chunk voxels. Maintained incrementally.

**Information gain**: Shannon entropy over unobserved voxels per chunk.

### Analytical Geological Prior (`multiagent/geological_prior.py`)

Precomputes `P(ore_type | y_sim, biome)` lookup table shape `(8, world_height, 5)` from `ORE_SPAWN_CONFIGS`. Handles MC→sim Y-coordinate conversion, triangle/uniform distributions, biome filtering, and cluster thresholds.

Methods: `query(y_sim, biome_id) → ndarray(8,)`, `query_chunk(cx, cz, biome_map) → ndarray(8,)` expected ore, `get_cluster_radius(ore_type) → float`.

### GNN Coordinator (`multiagent/coordinator/`)

**Heterogeneous graph** built every K=50 steps:

| Node Type | Features (dim) | Description |
|---|---|---|
| Agent (N_a) | 24 | position(3), fuel(1), inventory(8), current_target(3), steps_since_replan(1), preference(8) |
| Region (N_r) | 20 | center(3), expected_remaining(8), info_gain(1), explored_frac(1), biome_onehot(5), assigned_agents(1), bbox_mask(1) |

| Edge Type | Connectivity |
|---|---|
| agent → region | Fully connected bipartite |
| region → agent | Reverse of above |
| agent → agent | Pairs within 2×congestion_radius |
| region → region | 4-adjacent chunks |

**CoordinatorGNN**: HeteroConv wrapping GATv2Conv per edge type (3 layers, 4 heads, hidden=128). Score head: concat agent+region embeddings → MLP → scalar. Falls back to `CoordinatorGNNSimple` (MLP-only) when torch_geometric unavailable.

**Hungarian matching**: `scipy.optimize.linear_sum_assignment` on negated GNN scores. O(N³) — 100 agents < 10ms. Bounding box hard mask sets out-of-bounds regions to -inf.

**Task types** (`TaskType` enum): `MOVE_TO` (navigate), `EXCAVATE` (dig forward systematically), `MINE_ORE` (extract ore matching preference distribution), `RETURN_TO` (return to base).

**Replan triggers**: Every K=50 steps (periodic), or on event (ore discovery, chunk exhausted, agent join/leave, multiple PATH_BLOCKED).

### A* Pathfinder (`multiagent/pathfinding.py`)

6-connected grid A* (no diagonals — turtles can't move diagonally). Manhattan heuristic.

**Cost field**: `1.0` (air) + `dig_cost * is_solid` (3.0) + `congestion_cost * exp(-dist/radius)` near other agents (2.0, radius=3). Other agents are impassable.

**Action conversion**: `get_next_action()` maps path steps to turtle actions (FORWARD, UP, DOWN, TURN_LEFT/RIGHT, DIG/DIG_UP/DIG_DOWN). Handles dig-before-move via `_pending_dig_action`.

**Dynamic replan**: On PATH_BLOCKED, recompute from current position. Invalidate cached path if changed blocks are on the path.

### Shared World (`multiagent/shared_world.py`)

Wraps base `World` with multi-agent coordination:
- **Occupancy grid**: `np.int16` matching world shape. -1=empty, else agent_id. O(1) collision checks.
- **Agent registration/deregistration**, atomic `move_agent()` occupancy updates.
- **Telemetry buffer**: Per-agent event collection, `flush_telemetry()` returns and clears.
- **Agent spawning**: Randomized with min Manhattan distance constraint.
- **Agent density channel**: `get_agent_density_map()` for voxel observation channel 16.

### Multi-Agent Environment (`multiagent/agent/multi_agent_env.py`)

Per-agent `gym.Env` wrapping a `SharedWorld`:

**Observation space**:
- Voxels: `(16, FOG_Y, FOG_X, FOG_Z)` float16 — 15 existing channels + agent density channel
- Scalars: `(80,)` float32 — 70 existing dims + 10 task extras: `rel_target_xyz(3)`, `target_distance(1)`, `task_type_onehot(4)`, `distance_to_boundary(1)`, `inside_box_flag(1)`
- Preference: `(8,)` float32

**Hybrid navigation**: A* mode when distance > `mining_radius` (8 blocks), RL mode when close. A* provides action directly; RL policy sees full observation.

**Step flow**: Inspect 3 blocks → detect mismatches → choose action (A* or RL) → execute with occupancy check → emit telemetry → compute task reward → check completion.

### Multi-Agent Feature Extractor (`multiagent/agent/agent_policy.py`)

Extends single-agent `MiningFeatureExtractor`:
- 3D CNN: 16 input channels (15 base + agent density), 3 conv layers with FiLM conditioning + squeeze-excitation → 256-dim
- Scalar MLP: 80 input dims → 128 → 64-dim
- Output: concat [CNN(256), Scalars(64), Pref(8)] = **328-dim features**

### Task Rewards (`multiagent/agent/task_rewards.py`)

| Reward | Value | Condition |
|---|---|---|
| Task completion | +1.0 | Task-specific completion condition met |
| Progress toward goal | +0.1 | Potential-based (closer to target) |
| Block cleared | +0.05 | Any block mined (3x for target ore) |
| Regress from target | -0.02 | Moved away from goal |
| Idle (no useful action) | -0.01 | No block mined or progress |
| Inside bounding box | +0.05/step | Agent within assigned region |
| Outside bounding box | -0.2/step | Agent left assigned region |

**MINE_ORE distribution-aware rewards**: Per-block reward weighted by `preference[ore] * rarity_mult[ore]` (capped at 3.0). Non-target ores give `progress_reward * 0.3` (tolerant). `SpawnRateNormalizer` derives rarity multipliers from worldgen data so 1 diamond counts as much as N coal. Potential-based alignment bonus (`alignment_bonus * delta_cosine_similarity`) nudges mining toward the target distribution after `alignment_min_ores` (3) blocks mined. Per-agent preferences are blended from team preference and regional ore availability via `preference_blend_alpha` (0.5).

Task completion conditions: **MOVE_TO** — distance ≤ 1. **EXCAVATE** — blocks_cleared ≥ budget. **MINE_ORE** — coordinator decides (no auto-complete). **RETURN_TO** — same as MOVE_TO.

### Inter-Agent Communication (`multiagent/agent/communication.py`)

Message types: `ORE_FOUND`, `REGION_EMPTY`, `HAZARD`, `FUEL_LOW`. `MessageBuffer` with bounded inbox (max 50) and outbox for broadcast. Agents emit `ORE_FOUND` on discovering target ore.

### 4-Phase Training (`train_multiagent.py`)

| Phase | Description | Steps | Agents | Coordinator |
|---|---|---|---|---|
| 1 | Single-agent task mastery | 4M | 1 | TaskSampler (random tasks) |
| 2 | Multi-agent execution | 2M | 4→8→16→32 | Heuristic (belief map) |
| 3 | Coordinator GNN training | 2M | 32 | GNN (REINFORCE), agents frozen |
| 4 | Joint training | 3M | 32 | Blended: α lerp(heuristic, GNN) |

**Phase 1 task sampling** (`multiagent/training/task_sampler.py`): Weighted — MOVE_TO 35%, EXCAVATE 40%, MINE_ORE 20%, RETURN_TO 5%. MINE_ORE tasks use Dirichlet(0.5) preference vectors by default; other task types use one-hot. Curriculum difficulty: small→large boxes, short→long distances.

**Phase 2 agent count curriculum**: 4 → 8 → 16 → 32 agents (increase every 500K steps).

**Phase 3 REINFORCE** (`multiagent/training/coordinator_trainer.py`): GNN scores → log-softmax → gather matched pairs → policy gradient with EMA baseline. Entropy bonus (0.01) prevents assignment collapse. Grad clipping at 1.0.

**Phase 4 blended assignment**: α schedule [0.1, 0.3, 0.6, 1.0] stepped every 750K steps.

### Multi-Agent VecEnv (`multiagent/training/multi_agent_vec_env.py`)

Single-process `MultiAgentVecEnv` — all agents share one `SharedWorld` + `BeliefMap` + `Coordinator`. Each step: override A*-mode actions → step all envs → flush telemetry → update belief → replan if K steps elapsed. `num_envs = n_agents`.

### Multi-Agent Config (`multiagent/config.py`)

Key defaults:

| Category | Parameter | Default |
|---|---|---|
| Coordinator | `coordinator_interval_k` | 50 |
| GNN | `gnn_hidden_dim` / `layers` / `heads` | 128 / 3 / 4 |
| Belief map | `cluster_strength` / `cluster_radius` | 0.3 / 4.0 |
| A* | `dig_cost` / `congestion_cost` / `congestion_radius` | 3.0 / 2.0 / 3 |
| Navigation | `mining_radius` | 8 |
| Task rewards | completion / progress / block_cleared | 1.0 / 0.1 / 0.05 |
| Boundary | `box_stay_bonus` / `box_leave_penalty` | 0.05 / -0.2 |
| MINE_ORE | `alignment_bonus` / `rarity_mult_cap` | 0.05 / 5.0 |
| MINE_ORE | `alignment_min_ores` / `preference_blend_alpha` | 3 / 0.5 |
| Scale | `max_agents` | 100 |

### Config Extensions (`config.py`)

Added constants for multi-agent observation dimensions:
- `CH_AGENT_DENSITY = 15` — index of agent density voxel channel
- `SCALAR_OBS_DIM_MULTI = SCALAR_OBS_DIM + 10` (80) — task extras
- `CHUNK_SIZE_XZ = 16` — XZ chunk granularity for belief map

---

## Single-Agent Reward System

Two reward configurations exist: **Stage 1** (curriculum stage 0) and **Stages 2-5** (potential-based). Both produce four components (`r_harvest`, `r_adjacent`, `r_clear`, `r_ops`) that are scalarized via:

```
R = harvest_alpha * r_harvest + r_adjacent + r_clear + r_ops
```

### Stages 2-5: Potential-Based Harvest Efficiency (`RewardConfig`)

| Component | Parameter | Default | Formula / Justification |
|---|---|---|---|
| **Harvest delta** | `harvest_alpha` | `1.0` | Scales the per-step potential delta. Potential: `Phi(t) = sum_i[ w_i * (1 - exp(-mined_i / (kappa * ref + eps))) ]`. Delta = `Phi(t) - Phi(t-1)`. Exponential saturation gives diminishing returns per ore type. |
| | `harvest_kappa` | `0.4` | Saturation parameter — lower = faster saturation. Controls how quickly mining additional ores yields smaller deltas. |
| | `harvest_epsilon` | `1.0` | Prevents division by zero in the denominator `kappa * ref + eps`. |
| | `harvest_reference_total` | `400.0` | Fixed reference ore count for saturation denominator. Using a constant (not per-world) eliminates episode-to-episode reward variance. |
| **Maintenance bonus** | `potential_maintenance_bonus` | `0.005` | Per-step bonus = `0.005 * Phi(t)`. Rewards holding a high potential rather than only deltas, preventing the agent from stalling after initial harvest. |
| **Adjacent ore penalty** | `adjacent_penalty_beta` | `0.5` | Base penalty weight. Penalizes the agent for being next to desired ores without mining them. Formula: `-beta * tanh(raw_miss) * (1 + lambda * skip_count)`. |
| | `adjacent_skip_lambda` | `0.1` | Escalation per consecutive skip — the longer the agent ignores adjacent desired ores, the steeper the penalty. |
| | `adjacent_skip_cap` | `10` | Maximum consecutive skip count to bound the penalty escalation. |
| **Local clear bonus** | `local_clear_bonus` | `0.5` | One-time bonus when all adjacent desired ores are cleared (post-action weight drops to 0 from a positive pre-action weight). Rewards thorough local extraction. |
| **Fuel penalty** | `fuel_critical_threshold` | `0.2` | Fuel fraction below which the progressive penalty activates. |
| | `fuel_critical_penalty` | `-1.0` | Maximum penalty per step at fuel = 0. Ramps quadratically: `-1.0 * ((1 - fuel_frac / 0.2)^2)`. Replaces a hard death penalty with a smooth gradient. |
| **Coal refuel bonus** | `coal_refuel_bonus` | `0.1` (Stage 2) | Bonus added to `r_ops` when coal is mined for fuel. Scales with need: `bonus * (0.5 + 0.5 * (1 - fuel_frac))`. Range: 0.05 (full) to 0.10 (empty). Configured per-stage on `CurriculumStage`. |
| **Time penalty** | `time_penalty` | `-0.005` | Constant per-step cost. Encourages the agent to mine efficiently rather than wander. |

### Stage 1: Immediate Per-Ore Rewards (`Stage1RewardConfig`)

Stage 1 is the entry curriculum (dense world, infinite fuel, one-hot preference). Primary goal: learn to navigate to the correct Y-level for the target ore type and mine there.

| Component | Parameter | Default | Formula / Justification |
|---|---|---|---|
| **Harvest (per-ore)** | `per_ore_reward` | `5.0` | Immediate reward = `5.0 * pref[idx] * abundance_mult`. Abundance multiplier = `clamp(reference_count / target_count, 0.25, 10.0)` normalizes reward by world ore count. |
| | `harvest_reference_count` | `100.0` | Reference ore count for dynamic scaling. Worlds with fewer target ores give higher per-ore reward. |
| | `harvest_count_floor` | `0.25` | Minimum abundance multiplier (for very common ores). |
| | `harvest_count_ceil` | `10.0` | Maximum abundance multiplier (for very rare ores). |
| **Completion bonus** | `completion_scale` | `10.0` | Terminal bonus = `10.0 * (target_mined / target_in_world)`. End-of-episode incentive for maximizing target ore collection. |
| **Waste penalty** | `waste_beta` | `0.05` | Penalty for mining non-target blocks. Ramps as `-(0.05) * (waste_count / waste_ramp)^alpha`. Kept soft — depth navigation is the priority over waste avoidance. |
| | `waste_ramp` | `200` | Number of waste blocks over which the penalty ramps to full strength. Slow ramp lets agent build mining habit before refining. |
| | `waste_alpha` | `1.5` | Exponent for waste penalty ramp. |
| | `non_target_ore_multiplier` | `1.5` | Non-target *ores* count as 1.5x waste vs regular blocks (stone/dirt). |
| **Exploration bonus** | `exploration_bonus` | `0.002` | Small per-step reward for visiting a new cell. Decays with halflife = `exploration_decay_frac * volume`. |
| | `exploration_decay_frac` | `0.003125` | Fraction of world volume used as halflife. E.g. 40x40x40 → halflife=200. |
| **XZ exploration bonus** | `xz_exploration_bonus` | `0.03` | Per new XZ column bonus when at correct Y-depth: `0.03 / (1 + xz_count / halflife)`. Rewards horizontal spread at mining level; zero outside target Y-range. |
| | `xz_exploration_decay_frac` | `0.2` | Fraction of XZ area used as halflife. E.g. 40x40 → halflife=320. |
| **Y-distance penalty** | `y_penalty_scale` | `1.0` | Per-step cost when outside target ore's Y-range: `-(1.0) * (y_dist / world_height)^2 - y_penalty_base`. Quadratic scaling makes large distances much worse. Primary depth-navigation signal. |
| | `y_penalty_base` | `0.05` | Constant per-step cost for being off-depth (even by 1 block). |
| **Y-in-range bonus** | `y_in_range_bonus` | `0.0` | Per-step bonus when at the correct depth (disabled by default). |
| **Vertical progress** | `y_progress_scale` | `1.0` | Potential-based shaping: `scale * (prev_y_frac - curr_y_frac)`. Rewards movement toward target Y-range. Added to r_clear. |
| **Time penalty** | `time_penalty` | `-0.01` | Per-step cost to discourage idle looping. |
| **Idle penalty** | `idle_penalty_scale` | `-0.005` | Ramps with steps since last dig: `max(-0.5, -0.005 * (steps - grace))`. Grace period = 10 steps. |

### Scalarization

All four components feed into `PreferenceManager.scalarize()`:

```python
R = harvest_alpha * r_harvest + r_adjacent + r_clear + r_ops
```

- `r_harvest` is the only component scaled by `harvest_alpha` (default `1.0`)
- `r_adjacent` already incorporates its `-beta` factor
- `r_ops` values are already negative (fuel penalty + time penalty)
- Stage 1 adds `terminal_bonus` (completion bonus) at episode end

### Design Rationale

- **Potential-based shaping** (Stages 2-5): Provides dense reward signal from sparse ore mining events. The exponential saturation prevents reward hacking by over-mining a single ore type.
- **Fixed reference total**: Eliminates reward variance between worlds with different ore counts, stabilizing VecNormalize statistics.
- **Progressive fuel curve**: Smooth quadratic penalty replaces binary death, giving the value function a learnable gradient.
- **Adjacent penalty with decay**: Escalating skip penalty creates urgency around visible desired ores without harsh clipping.
- **All components in [-1, +1]**: Ensures stable PPO value function learning and VecNormalize statistics.

## Single-Agent Curriculum Stages

| Stage | Name | World Size | Ore Density | Fuel | Coal Refuel | Caves | Preference | Max Steps |
|---|---|---|---|---|---|---|---|---|
| 0 | `stage1_dense_easy` | 40x40x40 | 10x | Infinite | No | No | one_hot | 1000 |
| 1 | `stage2_sparse_fuel` | 32x64x32 | 3x (coal 1x) | 200 | 16/coal | No | one_hot | 2000 |
| 2 | `stage3_realistic_mixed` | 48x96x48 | 1x | 500 | No | No | two_mix | 1000 |
| 3 | `stage4_caves_dirichlet` | 64x128x64 | 1x | 800 | No | Yes | dirichlet | 1500 |
| 4 | `stage5_full` | 128x256x128 | 1x | 1000 | No | Yes | dirichlet | 2000 |

### Stage 2 Coal Refueling

Stage 2 introduces **coal-based fuel management**. The agent starts with 200 fuel (each movement costs 1). Mining coal ore restores 16 fuel (capped at max_fuel=200). Coal spawns at realistic 1x density while other ores use the 3x training multiplier via `ore_density_overrides`.

- **Refueling**: Applied in `mining_env.py step()` after dig, before termination check. Agent can mine coal at fuel=0 (digs are free) to survive.
- **Refuel bonus**: Small reward (0.05-0.10) added to `r_ops` when coal is mined. Scales with fuel need — higher bonus when fuel is lower.
- **Per-ore density**: `CurriculumStage.ore_density_overrides` dict overrides the global `ore_density_multiplier` for specific block types. Passed through `World` → `OreDistributor.place_ores()`.
