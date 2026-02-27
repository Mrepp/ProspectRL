# ProspectRL — CLAUDE.md

## Project Overview

PPO-based RL for ComputerCraft turtle mining in Minecraft. A turtle agent learns to navigate 3D voxel worlds and mine ores guided by a preference vector over 8 ore types.

## Quick Reference

- **Entry points**: `train.py` (training), `evaluate.py` (eval), `deployment/inference_server.py` (serve)
- **Config source of truth**: `config.py` — all hyperparameters, block defs, curriculum stages, reward settings
- **Reward logic**: `env/reward_vector.py` (component computation), `env/preference.py` (scalarization)
- **Tests**: `pytest` from repo root

## Reward System

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
| **Time penalty** | `time_penalty` | `-0.005` | Constant per-step cost. Encourages the agent to mine efficiently rather than wander. |

### Stage 1: Immediate Per-Ore Rewards (`Stage1RewardConfig`)

Stage 1 is the entry curriculum (dense world, infinite fuel, one-hot preference). Designed for fast initial learning.

| Component | Parameter | Default | Formula / Justification |
|---|---|---|---|
| **Harvest (per-ore)** | `per_ore_reward` | `1.0` | Immediate reward = `1.0 * preference[ore_idx]` when a target ore is mined. Direct signal — no potential shaping needed at this stage. |
| **Completion bonus** | `completion_scale` | `20.0` | Terminal bonus = `20.0 * (target_mined / target_in_world)`. Large end-of-episode incentive for maximizing target ore collection. |
| **Waste penalty** | `waste_beta` | `0.02` | Penalty for mining non-target blocks. Ramps as `-(0.02) * (waste_count / waste_ramp)^alpha`. Soft penalty — kept small in Stage 1 to avoid discouraging exploration. |
| | `waste_ramp` | `100` | Number of waste blocks over which the penalty ramps to full strength. |
| | `waste_alpha` | `2.0` | Exponent — quadratic ramp makes early waste nearly free but penalizes sustained non-target mining. |
| | `non_target_ore_multiplier` | `1.5` | Non-target *ores* count as 1.5x waste vs regular blocks (stone/dirt). Mining the wrong ore is worse than mining stone. |
| **Exploration bonus** | `exploration_bonus` | `0.01` | Small per-step reward for visiting a new cell. Encourages spatial coverage. |
| **Y-distance penalty** | `y_penalty_scale` | `0.03` | Per-step cost when outside target ore's Y-range. Scales linearly with distance: `-(0.03) * y_dist / world_height`. Guides the agent to the correct depth for its target ore. |

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

## Curriculum Stages

| Stage | Name | World Size | Ore Density | Fuel | Caves | Preference | Max Steps |
|---|---|---|---|---|---|---|---|
| 0 | `stage1_dense_easy` | 20x40x20 | 10x | Infinite | No | one_hot | 500 |
| 1 | `stage2_sparse_fuel` | 32x64x32 | 3x | 500 | No | one_hot | 800 |
| 2 | `stage3_realistic_mixed` | 48x96x48 | 1x | 500 | No | two_mix | 1000 |
| 3 | `stage4_caves_dirichlet` | 64x128x64 | 1x | 800 | Yes | dirichlet | 1500 |
| 4 | `stage5_full` | 128x256x128 | 1x | 1000 | Yes | dirichlet | 2000 |
