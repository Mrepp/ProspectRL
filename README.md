# ProspectRL

PPO-based reinforcement learning for ComputerCraft turtle mining in Minecraft. Supports **single-agent** training with curriculum learning and **multi-agent** coordination with 32-100+ turtles orchestrated by a Bayesian coordinator.

## Features

### Single-Agent
- **Preference-conditioned PPO** — a single model that adapts behavior based on a runtime preference vector over 8 ore types
- **Curriculum learning** — 5 stages from dense/easy to realistic cave worlds with fuel management
- **FiLM + Squeeze-Excitation** — 3D CNN feature extractor with FiLM conditioning on ore preference and SE attention
- **Action masking** — invalid actions (dig air, walk into walls, move without fuel) are masked at the policy level
- **Fog-of-war memory** — agents build spatial memory from limited observations, no omniscient world view

### Multi-Agent
- **Centralized Bayesian coordinator** — GNN-based task assignment with Hungarian matching for optimal agent→region allocation
- **Probabilistic belief map** — Bernoulli voxel model with Bayesian updates from telemetry, cluster propagation on ore discovery
- **Analytical geological prior** — P(ore|y, biome) derived from ore spawn configs for informed exploration
- **Congestion-aware A\* pathfinding** — 3D grid pathfinding with dig-through costs and repulsive fields near other agents
- **Hybrid A\*/RL navigation** — A\* for long-distance travel, RL policy for local mining within radius
- **4-phase training** — task mastery → multi-agent execution → coordinator training → joint optimization
- **Bounding box constraints** — user-defined spatial mining regions with learned boundary adherence

### Shared
- **3D world simulation** — procedural ore distribution and cave generation using OpenSimplex noise
- **FastAPI inference server** — deploy trained models with a REST API consumed by a Lua turtle client in-game
- **Real chunk evaluation** — test against actual Minecraft world data
- **Colab training support** — Jupyter notebook for training on free GPU instances

## Project Structure

```
prospect_rl/
├── config.py                  # All hyperparameters (single source of truth)
├── train.py                   # Single-agent training entrypoint
├── evaluate.py                # Single-agent evaluation
├── train_multiagent.py        # Multi-agent 4-phase training
├── evaluate_multiagent.py     # Multi-agent evaluation
├── env/                       # Single-agent Gymnasium environment
│   ├── mining_env.py          # Main env class
│   ├── reward_vector.py       # Reward computation
│   ├── preference.py          # Preference sampling + scalarization
│   ├── action_masking.py      # Valid action mask
│   ├── turtle.py              # Turtle state management
│   └── world/                 # World simulation
│       ├── world.py           # 3D grid manager
│       ├── ore_distribution.py
│       ├── cave_generation.py
│       └── noise_utils.py
├── models/                    # PPO + SB3 integration
│   ├── policy_network.py      # Custom feature extractor (FiLM+SE)
│   ├── ppo_config.py          # Env factory, VecNormalize, PPO setup
│   └── callbacks.py           # Checkpoint, curriculum, metrics
├── multiagent/                # Multi-agent coordination system
│   ├── config.py              # MultiAgentConfig dataclass
│   ├── telemetry.py           # Telemetry events from turtle observations
│   ├── belief_map.py          # Bernoulli voxel belief map + chunk tracking
│   ├── geological_prior.py    # Analytical P(ore|y,biome) prior
│   ├── pathfinding.py         # A* with congestion + dig-through
│   ├── shared_world.py        # SharedWorld with occupancy grid
│   ├── agent/                 # Per-agent environment and policy
│   │   ├── multi_agent_env.py # Per-agent env (A*/RL hybrid, telemetry)
│   │   ├── agent_policy.py    # MultiAgentFeatureExtractor (16ch, 80 scalars)
│   │   ├── task_rewards.py    # Task-specific reward computation
│   │   └── communication.py   # Inter-agent messaging
│   ├── coordinator/           # Centralized task assignment
│   │   ├── coordinator.py     # Graph build, GNN inference, Hungarian matching
│   │   ├── gnn.py             # CoordinatorGNN (HeteroConv + GATv2Conv)
│   │   └── assignment.py      # TaskAssignment, TaskType, BoundingBox
│   └── training/              # Multi-agent training infrastructure
│       ├── multi_agent_vec_env.py   # Single-process VecEnv
│       ├── coordinator_trainer.py   # REINFORCE for GNN
│       └── task_sampler.py          # Weighted task sampling
├── deployment/                # Inference server + Lua client
│   ├── inference_server.py    # Single-agent FastAPI server
│   └── lua/
│       └── turtle_client.lua  # CC:Tweaked Lua client
├── notebooks/                 # Colab training notebook
└── tests/                     # Mirrors source structure
    ├── ...                    # Single-agent tests
    └── multiagent/            # Multi-agent tests
```

## Installation

```bash
# Core (training)
pip install -e .

# With all extras (deploy + viz + dev)
pip install -e ".[all]"
```

Requires Python 3.10+. Multi-agent GNN coordinator requires `torch_geometric` (falls back to MLP-only mode without it).

## Usage

### Single-Agent Training

```bash
# Start training at curriculum stage 0
python train.py --stage 0

# Resume from checkpoint
python train.py --stage 0 --resume --checkpoint-dir ./checkpoints

# Custom settings
python train.py --stage 2 --n-envs 8 --total-timesteps 5000000
```

### Multi-Agent Training (4 Phases)

```bash
# Phase 1: Single-agent task mastery (learn MOVE_TO, EXCAVATE, MINE_ORE, RETURN_TO)
python train_multiagent.py --phase 1 --total-timesteps 4000000

# Phase 2: Multi-agent execution with heuristic coordinator
python train_multiagent.py --phase 2 --n-agents 32 --total-timesteps 2000000

# Phase 3: Train coordinator GNN (agents frozen)
python train_multiagent.py --phase 3 --n-agents 32 --total-timesteps 2000000

# Phase 4: Joint training with blended assignment
python train_multiagent.py --phase 4 --n-agents 32 --total-timesteps 3000000
```

### Evaluate

```bash
# Single-agent evaluation
python evaluate.py --model ./checkpoints/final/model.zip --episodes 20

# Multi-agent evaluation
python evaluate_multiagent.py --n-agents 32 --episodes 10 --max-steps 2000
```

### Deploy

```bash
# Start the inference server
python -m deployment.inference_server
```

The server exposes:
- `GET /health` — health check
- `POST /act` — get next action from observation
- `POST /preference` — generate a preference vector for a target ore

### Colab Training

Open `notebooks/colab_training.ipynb` in Google Colab for GPU-accelerated training.

## Multi-Agent Architecture

The multi-agent system uses a **centralized coordinator** with **decentralized execution**:

1. **Coordinator** (runs every K=50 steps): Builds a heterogeneous graph of agent and region nodes, runs GNN message passing, and uses Hungarian matching for optimal 1:1 agent→region task assignment.

2. **Belief Map**: Each voxel is a Bernoulli random variable updated via Bayesian inference from turtle telemetry. Cluster propagation boosts neighbor beliefs on ore discovery. Chunk-level tracking (UNKNOWN → EXPLORED → EXHAUSTED) guides exploration.

3. **Agents**: Navigate via A\* when far from targets (>8 blocks), switch to RL policy for local mining. Limited to 3-block observations per step (matching CC:Tweaked turtle hardware). Emit telemetry events that feed back to the coordinator.

4. **Task Types**: MOVE_TO (navigate), EXCAVATE (systematic mining), MINE_ORE (distribution-aware ore extraction with rarity-weighted rewards), RETURN_TO (return to base). Each includes a bounding box constraint for spatial coordination.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
