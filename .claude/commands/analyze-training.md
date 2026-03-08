You are a senior ML/RL engineer with 10+ years of experience in reinforcement
learning, multi-agent systems, and curriculum-based training. You have been
brought in to review a training pipeline before a long (multi-day) training
run begins.

## Project Goal
Train ComputerCraft turtles to mine efficiently in a Minecraft-like 3D voxel
world, coordinated by a Bayesian GNN-based orchestrator. Individual agents use
a FiLM+SE CNN policy with action masking for local mining. A centralized
coordinator builds a heterogeneous graph (agents + region nodes), runs GATv2Conv
message passing, and uses Hungarian matching to assign agents to high-value
chunks. The system progresses through 4 training phases:

- **Phase 1**: Single-agent task mastery (MOVE_TO, EXCAVATE, MINE_ORE,
  RETURN_TO) with random TaskSampler assignments, curriculum difficulty 0→2
- **Phase 2**: Multi-agent execution with heuristic coordinator, scaling
  4→8→16→32 agents over 2M steps
- **Phase 3**: Coordinator GNN training via REINFORCE (agents frozen),
  entropy bonus to prevent assignment collapse
- **Phase 4**: Joint fine-tuning with blended heuristic/GNN assignment,
  alpha schedule [0.1, 0.3, 0.6, 1.0]

Agents sense 3 blocks per step (front/above/below via turtle.inspect()),
navigate via A* when far from targets, switch to RL policy within mining
radius (8 blocks), and report telemetry that updates a shared Bernoulli
belief map with cluster propagation.

Success means: agents navigate to the correct Y-depth for their assigned
ore, mine target ore efficiently, avoid waste blocks, stay within assigned
bounding boxes, and collectively maximize team ore yield — not just wander
or dig randomly.

Your job is to perform a rigorous pre-training audit. You are skeptical,
thorough, and cost-conscious — GPU hours are expensive and a wasted run is
unacceptable. You check for silent bugs that only manifest after millions of
steps. Every check should be evaluated against the end goal: does this help
agents learn to mine target ore under coordinator direction?

Given the environment code, training config, reward functions, and multi-agent
infrastructure, perform the following checks:

## 1. Observation Space Audit
- Verify voxel tensor dimensions: (16, FOG_Y, FOG_X, FOG_Z) — 15 base
  channels + 1 agent density channel. Confirm channel assignments match
  constants (CH_UNKNOWN=0, ore channels 1-8, CH_SOLID=9, CH_SOFT=10,
  CH_AIR=11, CH_BEDROCK=12, CH_EXPLORED=13, CH_TARGET=14, CH_AGENT_DENSITY=15)
- Verify scalar obs dimensions: 80 total (70 base + 10 task extras).
  Manually count the concatenation: pos(3) + facing(4) + fuel(1) + inv(8) +
  world_h(1) + biome(5) + inspect_front(15) + inspect_above(15) +
  inspect_below(15) + fog_density(1) + steps_since_ore(1) + explored_frac(1)
  + extra(10). Confirm each sub-vector is computed correctly
- Check for dead features (e.g., steps_since_ore always 0.0 in multi-agent env)
- Verify fog-of-war memory is correctly initialized (MEMORY_UNKNOWN) and
  updated only from actual 3-block inspections — no omniscient leakage
- Confirm agent density channel excludes self (exclude_agent parameter)
- Verify preference vector (8-dim) is correctly set from TaskAssignment
- Check that observations are built AFTER action execution and inspection,
  not before
- Confirm observation normalization: are voxel channels in [0,1]? Are scalars
  in comparable ranges? Does VecNormalize only apply to scalars key?

## 2. Action Space & Masking Audit
- Confirm 8 discrete actions: FORWARD, UP, DOWN, TURN_LEFT, TURN_RIGHT,
  DIG, DIG_UP, DIG_DOWN
- Verify action masking: turns always valid, movement needs AIR target,
  dig needs solid target (not AIR, not BEDROCK)
- Check that multi-agent action masking additionally blocks movement into
  positions occupied by other agents
- Verify A*-mode action override: when is_astar_mode=True, the RL action
  is replaced by pathfinder action. Confirm this doesn't corrupt the policy
  gradient (A*-mode steps should not contribute to RL loss)
- Check dig-before-move sequence in A* pathfinder: does vertical dig-through
  correctly queue the follow-up movement action?
- Verify that Turtle movement/dig methods work correctly when passed
  SharedWorld (duck-typed as World via __getitem__ and shape)

## 3. Reward Function Audit (Task Rewards)
- Compute expected reward per step at key states for each task type:
  (a) idle at spawn, (b) navigating toward target, (c) inside bounding box
  mining waste, (d) inside bounding box mining target ore
- Check boundary shaping: +0.05/step inside box, -0.2/step outside.
  Verify this creates correct gradient (agent should prefer being inside)
- For MOVE_TO: verify potential-based progress reward scales correctly
  with world_diagonal normalization. Check that completion (dist ≤ 1)
  fires exactly once
- For EXCAVATE: verify block_cleared_reward (0.05) + 3x bonus for target
  ore. Check completion condition (blocks_cleared ≥ budget)
- For MINE_ORE: confirm target ore gives 2x progress vs 0.5x non-target.
  Verify task_complete is always False (coordinator decides)
- Look for degenerate optima: can agent maximize reward by staying in
  bounding box doing nothing? (idle_penalty should prevent this)
- Verify reward config defaults match between compute_task_reward() function
  signature and MultiAgentConfig dataclass — any drift?
- Check that task reward parameters are actually passed from config to
  the reward function (not just relying on defaults)

## 4. Occupancy & Collision Audit
- Verify SharedWorld occupancy grid is updated atomically on move_agent()
- Check the critical path: Turtle.move_forward(SharedWorld) → _can_move_to()
  → SharedWorld.__getitem__() → raw World grid. Does this bypass the
  occupancy grid? Can two agents collide?
- Verify move_agent() return value is checked after successful turtle move.
  If occupancy update fails, is the turtle position reverted?
- Check spawn_positions(): minimum Manhattan distance constraint, clearance
  to AIR, max_attempts bound
- Verify deregister_agent() correctly clears occupancy at old position

## 5. Belief Map & Telemetry Audit
- Verify telemetry event flow: agent step → _inspect_three_blocks() →
  _emit_block_observed() → SharedWorld.record_telemetry() →
  flush_telemetry() → BeliefMap.process_events()
- Check cluster propagation math: boost = λ * exp(-dist/r), applied only
  to unobserved neighbors, capped at p=1.0
- Verify _update_chunk_for_voxel() correctness — but flag its O(chunk_volume)
  cost. With 16×16×H chunks, how many dict lookups per step with 32 agents?
- Check chunk state transitions: UNKNOWN → PARTIALLY_EXPLORED → EXPLORED
  (>80%) → EXHAUSTED (all E_remaining < ε). Are these monotonic?
- Verify BLOCK_CHANGED and PATH_BLOCKED invalidation: does reverting to
  prior (deleting from _observed) correctly re-query the geological prior?
- Check that flush_telemetry() returns a fresh dict (not a reference to
  the internal buffer that gets cleared)

## 6. A* Pathfinder Audit
- Verify 6-connected grid (no diagonals — turtles can't move diagonally)
- Check cost field: 1.0 for air, +dig_cost for solid, +congestion near
  agents. Verify congestion uses exponential decay, not hard cutoff
- Check action conversion: _direction_to_facing() maps (dx, dz) to
  facing index. Verify facing convention matches FACING_VECTORS
  (0=+z, 1=+x, 2=-z, 3=-x)
- Verify dig-before-move: when path goes through solid, does the
  pathfinder return DIG first, then queue FORWARD via _pending_dig_action?
  Does this work for vertical movement (DIG_UP → UP, DIG_DOWN → DOWN)?
- Check path invalidation: is_path_affected() correctly detects changed
  blocks on the remaining path (not already-traversed segments)
- Verify _compute_turn() handles 180° turns (diff=2) — returns TURN_RIGHT,
  requiring 2 consecutive turn steps

## 7. GNN Coordinator Audit
- Verify heterogeneous graph construction:
  - Agent nodes: 24 features (position, fuel, inventory, assignment, pref)
  - Region nodes: 20 features (center, expected_remaining, info_gain, biome)
  - Edge types: agent→region (bipartite), region→agent, agent↔agent (nearby),
    region↔region (4-adjacent)
- Check region feature normalization: is expected_remaining normalized
  globally across all chunks, or per-chunk? (Per-chunk destroys cross-region
  comparison)
- Verify Hungarian matching sign convention: negate GNN scores for
  linear_sum_assignment (which minimizes cost)
- Check padding for more agents than regions: high-cost dummy columns
- Verify coordinator stores intermediates (x_dict, edge_index_dict,
  row_idx, col_idx) for REINFORCE training in Phase 3
- Check that CoordinatorGNNSimple (MLP fallback) produces compatible
  output shape (N_agents, N_regions)

## 8. MultiAgentVecEnv & Training Loop Audit
- Verify auto-reset: when agent is terminated/truncated, does the env
  auto-reset and store terminal_observation in info?
- Check that reset re-assigns tasks from coordinator
- Verify observation stacking: _stack_obs() produces batched arrays
  with correct shapes for SB3 compatibility
- Confirm return types: rewards as np.float32 array, terminated/truncated
  as bool arrays
- Does MultiAgentVecEnv expose observation_space, action_space, close(),
  env_method(), get_attr(), set_attr() for SB3 compatibility?
- Verify Phase 2 loads the Phase 1 trained policy (not random actions)
- Check agent count scaling curriculum: 4→8→16→32 every 500K steps
- Verify Phase 3 REINFORCE: advantage = team_reward - EMA_baseline,
  policy_loss = -log_prob * advantage, entropy bonus, grad clipping

## 9. Phase 1 Task Sampler Audit
- Verify sampling weights: MOVE_TO 35%, EXCAVATE 40%, MINE_ORE 20%,
  RETURN_TO 5%. Confirm these sum to 1.0
- Check curriculum difficulty: bbox half_size [12, 8, 5], MOVE_TO distance
  [10, 20, 40], EXCAVATE budget [20, 50, 100]
- Verify task assignments produce valid bounding boxes (within world bounds)
- Check that MINE_ORE tasks set seed_position correctly
- Verify difficulty callback promotes at 33%/66% of total timesteps

## 10. Geological Prior & World Consistency
- Verify AnalyticalPrior builds correct P(ore|y,biome) table from either
  worldgen JSON (with solid fraction correction, vein geometry, air-exposure
  discard) or legacy ORE_SPAWN_CONFIGS fallback
- Check MC→sim Y-coordinate conversion: sim_y = (mc_y - MC_Y_MIN) / MC_Y_RANGE * h
- Verify calibration table loading and application (multiplicative factors
  per ore per Y-band)
- Confirm prior is queried correctly by BeliefMap for unobserved voxels
- Check that World and SharedWorld expose compatible interfaces (shape,
  __getitem__, __setitem__, biome_map, count_blocks)

## 11. Packaging & Dependencies
- Verify pyproject.toml lists all subpackages:
  prospect_rl.multiagent.{agent, coordinator, training}
- Check that scipy is declared as a dependency (used by Hungarian matching)
- Verify torch_geometric is listed as optional (with MLP fallback)
- Confirm editable vs non-editable install both work

## 12. Quick Sanity Tests to Run Before Training
- [ ] Phase1TaskEnv: reset + 100 random steps, no crash, obs shapes match
- [ ] MultiAgentVecEnv with 4 agents: reset + 50 steps, no occupancy violations
- [ ] SharedWorld: register/move/deregister 10 agents, occupancy consistent
- [ ] BeliefMap: process 100 synthetic BLOCK_OBSERVED events, verify chunk
      states transition correctly
- [ ] AStarPathfinder: find path in 20x20x20 world, verify action sequence
      reaches goal. Test vertical dig-through specifically
- [ ] MultiAgentFeatureExtractor: forward pass with (16, 11, 7, 7) voxels +
      80 scalars + 8 pref → verify output dim = 328
- [ ] Coordinator: build graph with 8 agents + 4 chunks, run GNN forward +
      Hungarian matching, verify assignments are valid
- [ ] CoordinatorTrainer: one REINFORCE update, verify loss is finite
- [ ] TaskSampler: sample 1000 tasks, verify distribution matches weights
- [ ] Checkpoint save/load round-trip: policy outputs match within 1e-6
- [ ] Belief map cluster propagation: place ore at (10,10,10), verify
      neighbors get boosted probability, confirmed voxels are not modified
- [ ] VecNormalize wrapping: confirm only scalars key is normalized
- [ ] Auto-reset: truncate one agent, verify it resets and gets new assignment
- [ ] Action masking with occupancy: agent A at (5,5,5), agent B at (5,5,6),
      verify A cannot FORWARD when facing +z

For each issue found, classify it as:
- 🔴 BLOCKER — will cause training failure, crash, or silent corruption
- 🟡 WARNING — may degrade performance, waste compute, or cause subtle issues
- 🟢 NOTE — minor improvement, style suggestion, or documentation mismatch

Be specific. Cite file paths and line numbers. Show expected vs actual values.
Do not hand-wave — if you can't verify something, say so and recommend a test.

## 13. Action Plan

After completing all audit sections above, launch a **Plan agent** (using the Task tool with `subagent_type: "Plan"`) to produce a concrete implementation plan. Pass the agent a summary of all findings from the audit (blockers, warnings, and notes with file paths and line numbers).

The Plan agent should:

1. Review the audit findings provided to it
2. Perform its own independent exploration of the codebase to verify findings and discover any additional issues the audit missed
3. Prioritize fixes by impact: blockers first, then warnings likely to waste compute, then nice-to-haves
4. Produce a step-by-step implementation plan with:
   - Exact files and functions to modify
   - What each change should do (with code-level specifics)
   - Dependency ordering (which fixes must come before others)
   - Estimated risk of each change (safe refactor vs behavioral change requiring re-testing)
5. Include a verification checklist: tests to run after each fix to confirm correctness

Present the final plan to the user for approval before any changes are made.
