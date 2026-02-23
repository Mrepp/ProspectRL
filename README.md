# ProspectRL

PPO-based reinforcement learning for ComputerCraft turtle mining in Minecraft. A turtle agent learns to navigate 3D voxel worlds and mine ores guided by a user-specified **preference vector** — tell it to prioritize diamonds, iron, or any mix of ores.

## Features

- **Preference-conditioned PPO** — a single model that adapts behavior based on a runtime preference vector over 7 ore types
- **Curriculum learning** — 5 stages from dense/easy to realistic cave worlds with fuel management
- **Action masking** — invalid actions (dig air, walk into walls, move without fuel) are masked at the policy level
- **3D world simulation** — procedural ore distribution and cave generation using OpenSimplex noise
- **FastAPI inference server** — deploy trained models with a REST API consumed by a Lua turtle client in-game
- **Colab training support** — Jupyter notebook for training on free GPU instances

## Project Structure

```
prospect_rl/
├── config.py              # All hyperparameters (single source of truth)
├── train.py               # CLI training entrypoint
├── evaluate.py            # Evaluation across preference grid
├── env/                   # Gymnasium environment
│   ├── mining_env.py      # Main env class
│   ├── reward_vector.py   # Reward computation
│   ├── preference.py      # Preference sampling + scalarization
│   ├── action_masking.py  # Valid action mask
│   ├── turtle.py          # Turtle state management
│   └── world/             # World simulation
│       ├── world.py       # 3D grid manager
│       ├── ore_distribution.py
│       ├── cave_generation.py
│       └── noise_utils.py
├── models/                # PPO + SB3 integration
│   ├── policy_network.py  # Custom feature extractor
│   ├── ppo_config.py      # Env factory, VecNormalize, PPO setup
│   └── callbacks.py       # Checkpoint, curriculum, metrics
├── deployment/            # Inference server + Lua client
│   ├── inference_server.py
│   ├── model_export.py
│   └── lua/turtle_client.lua
├── notebooks/             # Colab training notebook
└── tests/                 # Mirrors source structure
```

## Installation

```bash
# Core (training)
pip install -e .

# With all extras (deploy + viz + dev)
pip install -e ".[all]"
```

Requires Python 3.10+.

## Usage

### Train

```bash
# Start training at curriculum stage 0
python -m prospect_rl.train --stage 0

# Resume from checkpoint
python -m prospect_rl.train --stage 0 --resume --checkpoint-dir ./checkpoints

# Custom settings
python -m prospect_rl.train --stage 2 --n-envs 8 --total-timesteps 5000000
```

### Evaluate

```bash
# Evaluate across all preference vectors
python -m prospect_rl.evaluate --model ./checkpoints/final/model.zip --episodes 20
```

### Deploy

```bash
# Start the inference server
python -m prospect_rl.deployment.inference_server
```

The server exposes:
- `GET /health` — health check
- `POST /act` — get next action from observation
- `POST /preference` — generate a preference vector for a target ore

### Colab Training

Open `notebooks/colab_training.ipynb` in Google Colab for GPU-accelerated training.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check prospect_rl/
```

## License

MIT
