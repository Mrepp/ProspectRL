Analyze ProspectRL PPO training logs and diagnose training issues.

ProspectRL trains RL agents for multi-agent cooperative mining in Minecraft, where agents learn to navigate 3D voxel worlds and mine ores guided by a tunable preference vector over 8 ore types. The long-term goal is teams of agents that can be configured via preferences to target specific ores efficiently.

If the user provided training log data as arguments ($ARGUMENTS), use that data. Otherwise, ask the user to paste their training metrics (TensorBoard scalars, stdout logs, or CSV dumps).

## Step 1: Discover and Read the Codebase

File paths shift between commits. Do NOT hardcode paths — discover them dynamically.

### 1a. Discover all Python source files

Run: `find . -name "*.py" -not -path "*/__pycache__/*" -not -path "*/.venv/*" -not -path "*/venv/*" | sort`

This gives you the ground truth of what exists right now.

### 1b. Read all critical files (in parallel where possible)

Read EVERY file in the following categories. If a file doesn't exist at the expected path, search for it using the file list from 1a. The codebase is small enough to read entirely — do NOT skip files.

**Reward system (how reward signals are computed):**
- Find and read the file containing `Stage1RewardConfig` and `RewardConfig` dataclasses (currently `config.py`). These define ALL reward hyperparameter defaults.
- Find and read the file containing `compute_stage1_reward_components()` and `compute_reward_components()` (currently `env/reward_vector.py`). This is the core reward math.
- Find and read the file containing `PreferenceManager` and `scalarize()` (currently `env/preference.py`). This shows how the 4 reward components are combined into a single scalar.

**Environment (how the agent interacts with the world):**
- Find and read the main environment file containing `MinecraftMiningEnv` (currently `env/mining_env.py`). Pay close attention to:
  - `reset()` — how state is initialized, how preference is sampled, how `_prev_y_dist` and other shaping state is set
  - `step()` — how reward components from `compute_stage1_reward_components()` are augmented with spin/loiter/noop/idle penalties in `r_ops`, the Y-arrival bonus added to `r_clear`, and how terminal bonus is added
  - `_build_obs()` and `_build_voxel_tensor()` — what the agent actually sees (channel layout, scalar features, target ore highlight channel)
  - `_nearest_target_ore_dist()` — used for approach bonus shaping
- Find and read the action masking file (currently `env/action_masking.py`) — which actions are masked and why (movement needs AIR, dig needs solid, turns always valid)
- Find and read the turtle state file (currently `env/turtle.py`) — movement/dig mechanics, fuel consumption, inventory tracking

**Model architecture (what the neural network looks like):**
- Find and read the feature extractor file (currently `models/policy_network.py`). Understand:
  - The 3D CNN architecture (conv layers, FiLM conditioning from preference vector, squeeze-excitation attention)
  - How voxel, scalar, and preference branches are combined
  - Output dimensionality of each branch and total features_dim
- Find and read the PPO model factory (currently `models/ppo_config.py`). Understand:
  - `make_training_env()` — how environments are created, VecNormalize setup (`norm_obs_keys=["scalars"]`), deterministic ore-to-env assignment (`fixed_ore_index = i % NUM_ORE_TYPES`)
  - `create_ppo_model()` — how MaskablePPO is configured, learning rate schedule (linear decay), policy_kwargs, net_arch for pi and vf heads
- Find and read the PPO hyperparameters: `PPOConfig` dataclass (currently in `config.py`) — learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm, pi/vf net_arch

**Training orchestration:**
- Find and read `train.py` — how training is launched, callback setup, checkpoint resume logic
- Find and read the callbacks file (currently `models/callbacks.py`). Read `MetricsCallback` carefully to confirm ACTUAL metric names logged to TensorBoard — these names change between versions.

**World generation (affects ore availability and training difficulty):**
- Find and read the world file (currently `env/world/world.py`) — generation pipeline order (stone → bedrock → biome → deepslate → fillers → ores → caves)
- Find and read ore distribution (currently `env/world/ore_distribution.py`) — how `OreSpawnConfig` parameters translate to actual ore placement probability, spatial clustering, biome restrictions
- Find and read biome generation (currently `env/world/biome.py`) — how biome map is generated, which biomes exist, noise thresholds
- Find and read `ORE_SPAWN_CONFIGS` and `FILLER_SPAWN_CONFIGS` (currently in `config.py`) — the actual per-ore spawn parameters

**Curriculum system:**
- Find and read `CURRICULUM_STAGES` (currently in `config.py`) — world size, ore density multiplier, fuel, caves, preference mode, max steps, advancement criteria per stage

**Observation space:**
- Find and read observation constants: `OBS_WINDOW_*`, `NUM_VOXEL_CHANNELS`, `SCALAR_OBS_DIM`, channel index constants (`CH_SOLID`, `CH_TARGET`, etc.) — currently in `config.py`

### 1c. Build a mental model

After reading, you should be able to answer these questions (verify each against the code you read):

1. **Reward assembly**: What is the exact per-step reward formula? Trace from `compute_stage1_reward_components()` return values through `mining_env.py:step()` (where spin/loiter/noop/idle are added to r_ops, y_arrival_bonus to r_clear) through `PreferenceManager.scalarize()` (harvest_alpha * r_harvest + r_adjacent + r_clear + r_ops) plus terminal_bonus.

2. **Action masking**: Which actions can be masked? (movement needs AIR target, dig needs solid target, turns always valid). How does this interact with noop_penalty? (noop fires on failed movement, but masked actions shouldn't be selected — so noop only fires if masking has a bug or if the agent is against a world boundary)

3. **Observation space**: What does the agent see? (14-channel 3D voxel tensor with per-ore channels + solid/soft/air/bedrock/explored/target, 17-dim scalar vector with normalized position + facing one-hot + fuel fraction + inventory counts + world height, 8-dim preference vector)

4. **VecNormalize**: What is normalized? (Only "scalars" key — voxels and pref are NOT normalized. Reward IS normalized.)

5. **Ore availability**: For the current curriculum stage, how many target ores actually exist in the world? This depends on world_size, ore_density_multiplier, the specific OreSpawnConfig parameters, and biome forcing.

6. **Agent spawn**: Where does the agent start? (center X/Z, randomized Y within ±25% of world center). How far is this from typical target ore Y-ranges?

7. **Model capacity**: Is the network big enough? (3-layer 3D CNN with 32→64→128 channels + FiLM + SE, FC to 256, scalar MLP to 32, total 296 features → pi_net [64,64] and vf_net [128,128])

## Step 2: Identify Training Stage

Determine the curriculum stage from the logs. Look for:
- `curriculum/stage` metric (0-4)
- Stage-specific metrics (e.g. `target_ores_mined` and `completion_ratio` for Stage 1, `harvest_potential` for later stages)
- Cross-reference metric names against what `MetricsCallback` actually logs — names evolve as the codebase changes

## Step 3: Parse and Categorize Metrics

Organize all available metrics into these categories:

**Mining Performance:**
- `target_ores_mined` — target ores per episode (Stage 1). Zero = agent not finding/mining target.
- `completion_ratio` — target_mined / total_in_world. Zero = no progress.
- `cumulative_waste` — non-target blocks mined (Stage 1). High waste + zero target = indiscriminate mining.
- `ores_per_episode` — all ores mined (including non-target).
- `blocks_per_episode` — total blocks mined (ores + stone/dirt).
- `fuel_efficiency` — ores per step.
- `harvest_potential` — cumulative potential value (Stages 2-5).

**Navigation:**
- Y-penalty metric (check actual name in callbacks — Stage 1 uses `cumulative_y_penalty`, Stages 2-5 use `cumulative_adjacent_penalty`) — total Y-distance penalty. Large negative = never reaching correct depth.
- `explored_count` — unique 3D cells visited. For 40x40x40 world: <100 stuck, 200-400 decent, >500 good.
- `episode_steps` — if always = max_steps (1000 for Stage 1), episodes always time out.

**PPO Diagnostics:**
- `entropy_loss` — policy randomness. Healthy: 0.5–1.5 for 8 actions (max entropy = ln(8) ≈ 2.08). <0.3 collapsed, >1.8 too random.
- `approx_kl` — policy change per update. Healthy: 0.005–0.02. >0.03 unstable, <0.001 not learning.
- `explained_variance` — value function quality. >0.5 good, <0.2 broken, negative = worse than mean.
- `clip_fraction` — PPO clipping rate. Healthy: 0.05–0.2. >0.3 learning rate too high.
- `policy_gradient_loss` — should be negative.
- `value_loss` — should decrease over time.

**Reward Components:**
- `cumulative_harvest_delta` — sum of r_harvest per episode. Zero = no target ores mined.
- `terminal_bonus` — end-of-episode completion bonus (Stage 1).
- `local_clears` — count of local-clear events (Stages 2-5).

## Step 4: Diagnose Issues

Check for these failure patterns IN ORDER OF SEVERITY:

### Critical (training is broken)
1. **Entropy collapse** (`entropy_loss < 0.3`): Policy locked into one action. Penalties dominate rewards. Fix: raise `ent_coef`, reduce penalty magnitudes, increase `clip_range`.
2. **Value function failure** (`explained_variance < 0`): Value predictions anti-correlated with returns. Fix: check reward budget balance, reduce reward variance, consider lower `gamma`.
3. **KL explosion** (`approx_kl > 0.05`): Policy updates too large. Fix: reduce `learning_rate`, reduce `clip_range`, increase `batch_size`.

### Major (agent not learning the right thing)
4. **Zero target ores** (`target_ores_mined=0`, `cumulative_harvest_delta=0`): Agent mines but never hits target. Check:
   - Y-penalty magnitude: large negative → agent not at correct depth → increase `y_penalty_scale`, `y_progress_scale`, `y_arrival_bonus`
   - Small Y-penalty but still zero target → agent at depth but not mining right blocks → increase `per_ore_reward`, check abundance multiplier for rare ores
5. **High waste, zero target** (`cumulative_waste >> 0`, `target_ores_mined=0`): Agent mines indiscriminately. Ore targeting signal too weak vs mining incentives. Fix: increase `per_ore_reward`, possibly increase `waste_beta` (but only after agent learns depth).
6. **Agent not mining** (`blocks_per_episode < 10`): Agent moves but doesn't dig. May be stuck in movement loops or penalties make "do nothing" optimal. Check reward budget — if net per-step reward is negative for a random-but-moving agent, the agent learns to minimize actions.

### Moderate (learning but suboptimal)
7. **Low exploration** (`explored_count < 100` in 40x40x40): Agent stuck in small area. Fix: increase `loiter_penalty`, `exploration_bonus`, `xz_exploration_bonus`.
8. **High clip fraction** (`clip_fraction > 0.3`): Fix: reduce `learning_rate`, increase `n_steps`.
9. **Entropy too high** (`entropy_loss > 1.8` after 100k+ steps): Reward signal too weak or noisy. Check reward budget, reduce `ent_coef`.
10. **Always timeout** (`episode_steps = max_steps` consistently past 200k steps): Task too hard or reward too sparse.

### Architecture-specific issues
11. **FiLM not conditioning**: If agent ignores preference (mines same ore regardless of pref vector), the FiLM layers may not be learning useful modulation. Check if preference-conditioned eval shows similar performance across all ore types. The FiLM init is near-identity (gamma≈1, beta≈0) which is conservative — may need more training steps.
12. **VecNormalize reward clipping**: Rewards are clipped to ±10.0. If raw rewards regularly exceed this (e.g., large y_arrival_bonus + per_ore_reward in same step), the agent sees clipped values and loses gradient information.
13. **Linear LR schedule**: Learning rate decays linearly to 0. If training for <1M steps, the LR may drop too fast. Check if `approx_kl` is dropping to near-zero late in training.

## Step 5: Reward Budget Analysis

This is the most important diagnostic. Compute approximate per-step reward budget using ACTUAL config values read from the source files in Step 1.

**Stage 1 positive signals:**
- Per target ore: `per_ore_reward * pref * abundance_mult` (amortized: multiply by expected ores mined / max_steps)
- Y-arrival bonus: `y_arrival_bonus` (one-time, added to r_clear, amortize over steps to reach depth)
- Exploration: `exploration_bonus / (1 + count/halflife)` per new cell
- XZ exploration: `xz_exploration_bonus / (1 + count/halflife)` per new column at depth (only when y_min <= turtle_y <= y_max)
- Y-progress: `y_progress_scale * (prev_frac - curr_frac)` per step moving toward target Y
- Approach bonus: `approach_bonus_scale * (prev_dist - curr_dist)` when target ore is visible in obs window
- Terminal bonus: `completion_scale * ratio` (episode end only)

**Stage 1 negative signals (every step):**
- Time: `time_penalty` (constant, in r_ops from compute fn)
- Y-distance: `-y_penalty_scale * (dist/height)^2 - y_penalty_base` (off-depth, in r_adjacent)
- Idle: `idle_penalty_scale * max(0, steps_since_dig - grace)` (capped at -0.5, added to r_ops in step())
- Loiter: `loiter_penalty` (when <unique_threshold unique positions in loiter_window-step window, added to r_ops in step())
- Waste: `-waste_beta * (count/ramp)^alpha` (added to r_ops in compute fn)
- Spin: `spin_penalty` (3+ consecutive same turns, added to r_ops in step())
- Noop: `noop_penalty` (failed movement, added to r_ops in step())

**Scalarization**: `R = harvest_alpha * r_harvest + r_adjacent + r_clear + r_ops + terminal_bonus`

Note: `r_ops` accumulates values from BOTH the compute function (time_penalty + waste) AND mining_env.py step() (spin + loiter + noop + idle). Watch for assignment-vs-accumulation bugs.

**If net per-step reward is negative for a "random but moving" agent, the agent learns to minimize actions. This is the #1 training failure mode.**

Compute approximate budgets for: (a) random agent, (b) agent at correct depth mining randomly, (c) ideal agent mining target. The gaps between these tell you how learnable the reward landscape is.

Also consider:
- **Reward variance**: Large one-time bonuses (y_arrival, terminal) create high variance in episode returns, making the value function harder to learn. Compare one-time bonus magnitude to cumulative per-step reward.
- **Reward delay**: How many steps before the agent gets its first positive signal? If it takes 50+ steps of negative reward before any positive, the credit assignment problem is severe.
- **Action masking impact**: With action masking, the effective action space is often 3-5 actions (not 8). This means max entropy is lower, so entropy_loss thresholds should be adjusted. Healthy entropy with masking may be 0.4-1.2.

## Step 6: Model Architecture Analysis

Check if the model architecture is appropriate for the current training stage:

- **Feature extractor capacity**: The 3D CNN processes a 14×32×9×9 tensor. With stride-2 convolutions, the spatial dimensions shrink to ~4×2×2 by layer 3. Verify the CNN isn't throwing away too much spatial information.
- **FiLM conditioning strength**: FiLM layers modulate CNN features based on preference. If training is stuck on Stage 1 (one-hot preferences), FiLM may not get diverse enough conditioning signals to learn useful modulation.
- **Policy/value head separation**: Separate net_arch for pi and vf allows different capacity. If value_loss is high but policy is learning, vf may need more capacity (or vice versa).
- **Total parameter count**: Estimate from architecture. If the model is too large for the data throughput (n_envs * n_steps per update), it may overfit within each rollout.

## Step 7: Output Structured Analysis

Format your response as:

### Training Diagnosis: [Stage Name]

**Summary**: [1-2 sentence overview of what's happening and how it relates to the goal of training preference-conditioned mining agents]

**Codebase Snapshot** (from files read in Step 1):
| Parameter | Value | Source File |
|---|---|---|
| [key reward/PPO params] | [actual value] | [file:line] |

**Red Flags**:
- [Bullet list of most concerning metrics with values]

**Issue #1: [Name]** (Severity: Critical/Major/Moderate)
- What: [What the metric shows]
- Why: [Root cause based on reward logic, model architecture, and config values — cite specific code paths]
- Fix: In `[file_path]`, change `param` from `current_value` to `suggested_value` — [reason]

[Continue for top 3-5 issues]

**Reward Budget**:
| Signal | Per-Step Value | Notes |
|---|---|---|
| [component] | [value] | [amortized or direct, cite formula from source] |
| **Net per step (random agent)** | **[sum]** | |
| **Net per step (at depth, mining randomly)** | **[sum]** | |
| **Net per step (ideal agent)** | **[sum]** | |

**Architecture Notes**:
- [Any concerns about model capacity, FiLM conditioning, obs space issues]

**Recommended Changes** (priority order):
1. In `[file_path]` `[class/function]`: change `param` from `X` to `Y` — [reason with reference to reward budget or architecture analysis]
2. ...

**What to Monitor After Changes**:
- [metric to watch and expected direction]
