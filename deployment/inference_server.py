"""FastAPI inference server for the ProspectRL agent.

Loads a trained MaskablePPO model and VecNormalize statistics on startup,
then exposes endpoints for health checks, action prediction, and preference
vector generation.

Start with::

    python -m prospect_rl.deployment.inference_server
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from prospect_rl.config import (
    BLOCK_NAME_TO_ID,
    CH_AIR,
    CH_BEDROCK,
    CH_EXPLORED,
    CH_SOFT,
    CH_SOLID,
    MAX_WORLD_HEIGHT,
    NUM_ORE_TYPES,
    NUM_VOXEL_CHANNELS,
    OBS_WINDOW_RADIUS_XZ,
    OBS_WINDOW_X,
    OBS_WINDOW_Y,
    OBS_WINDOW_Y_BELOW,
    OBS_WINDOW_Z,
    ORE_TYPES,
    SOFT_BLOCKS,
    SOLID_BLOCKS,
    Action,
    BlockType,
    DeploymentConfig,
)
from pydantic import BaseModel, Field

logger = logging.getLogger("prospect_rl.server")

# Window volume (X * Y * Z)
_WINDOW_VOLUME = OBS_WINDOW_X * OBS_WINDOW_Y * OBS_WINDOW_Z  # 9*32*9 = 2592

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ActRequest(BaseModel):
    """Observation payload sent by the Lua turtle client."""

    position: list[float] = Field(..., min_length=3, max_length=3)
    facing: int = Field(..., ge=0, le=3)
    fuel: int = Field(..., ge=0)
    inventory: dict[str, int] = Field(default_factory=dict)
    nearby_blocks: list[str] = Field(
        ...,
        min_length=_WINDOW_VOLUME,
        max_length=_WINDOW_VOLUME,
    )
    preference: list[float] = Field(
        ...,
        min_length=NUM_ORE_TYPES,
        max_length=NUM_ORE_TYPES,
    )
    world_size: list[int] = Field(
        default=[64, 128, 64], min_length=3, max_length=3,
    )
    max_fuel: int = Field(default=1000, ge=1)


class ActResponse(BaseModel):
    action: int
    action_name: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PreferenceRequest(BaseModel):
    target: str


class PreferenceResponse(BaseModel):
    preference: list[float]


# ---------------------------------------------------------------------------
# Global model holder
# ---------------------------------------------------------------------------

_model: Any = None
_vec_normalize: Any = None


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

# Ore-name order for preference vector (matches ORE_TYPES index)
_ORE_NAME_ORDER: list[str] = [
    "coal",
    "iron",
    "gold",
    "diamond",
    "redstone",
    "emerald",
    "lapis",
    "copper",
]

# Build ore-name -> preference-index mapping
_ORE_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(_ORE_NAME_ORDER)}

# Pre-compute arrays for block-to-channel mapping
_SOLID_ARRAY = np.array(sorted(SOLID_BLOCKS), dtype=np.int8)
_SOFT_ARRAY = np.array(sorted(SOFT_BLOCKS), dtype=np.int8)


def _build_observation(req: ActRequest) -> dict[str, np.ndarray]:
    """Convert an ActRequest into the Dict observation expected by the model.

    Returns ``{voxels, scalars, pref}`` matching ``mining_env._build_obs``.
    """
    ws = np.array(req.world_size, dtype=np.float32)
    pos = np.array(req.position, dtype=np.float32)

    # --- Voxel tensor (C, Y, X, Z) ---
    # Convert flat block name list to 3D int8 array (X, Y, Z)
    block_ids = np.array(
        [BLOCK_NAME_TO_ID.get(b, BlockType.STONE) for b in req.nearby_blocks],
        dtype=np.int8,
    ).reshape((OBS_WINDOW_X, OBS_WINDOW_Y, OBS_WINDOW_Z))

    # Transpose to (Y, X, Z) for channel encoding
    raw_yxz = block_ids.transpose(1, 0, 2)

    tensor = np.zeros(
        (NUM_VOXEL_CHANNELS, OBS_WINDOW_Y, OBS_WINDOW_X, OBS_WINDOW_Z),
        dtype=np.float32,
    )

    # Per-ore channels (0 .. NUM_ORE_TYPES-1)
    for i, ore_bt in enumerate(ORE_TYPES):
        tensor[i] = (raw_yxz == int(ore_bt)).astype(np.float32)

    # Grouped channels
    tensor[CH_SOLID] = np.isin(raw_yxz, _SOLID_ARRAY).astype(np.float32)
    tensor[CH_SOFT] = np.isin(raw_yxz, _SOFT_ARRAY).astype(np.float32)
    tensor[CH_AIR] = (raw_yxz == int(BlockType.AIR)).astype(np.float32)
    tensor[CH_BEDROCK] = (raw_yxz == int(BlockType.BEDROCK)).astype(np.float32)

    # Explored mask — in deployment all positions are considered explored
    tensor[CH_EXPLORED] = 1.0

    # --- Scalar features ---
    norm_pos = pos / np.maximum(ws, 1.0)

    facing_oh = np.zeros(4, dtype=np.float32)
    facing_oh[req.facing] = 1.0

    fuel = np.array(
        [float(req.fuel) / max(req.max_fuel, 1)],
        dtype=np.float32,
    )

    inv = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
    for name, count in req.inventory.items():
        idx = _ORE_NAME_TO_INDEX.get(name)
        if idx is not None:
            inv[idx] = float(count)

    world_height_norm = np.array(
        [ws[1] / MAX_WORLD_HEIGHT], dtype=np.float32,
    )

    scalars = np.concatenate([
        norm_pos, facing_oh, fuel, inv, world_height_norm,
    ])

    pref_vec = np.array(req.preference, dtype=np.float32)

    return {"voxels": tensor, "scalars": scalars, "pref": pref_vec}


def _build_action_mask(req: ActRequest) -> np.ndarray:
    """Compute a simple action mask from the request state.

    Uses the centre of the sliding window to determine movement/dig validity.
    The turtle is at window position (radius_xz, y_below, radius_xz).
    """
    mask = np.ones(len(Action), dtype=bool)

    facing_vecs = {
        0: np.array([0, 0, 1]),
        1: np.array([1, 0, 0]),
        2: np.array([0, 0, -1]),
        3: np.array([-1, 0, 0]),
    }
    fv = facing_vecs[req.facing]

    # Reconstruct block IDs in (X, Y, Z) layout
    block_ids = np.array(
        [BLOCK_NAME_TO_ID.get(b, BlockType.STONE) for b in req.nearby_blocks],
        dtype=np.int8,
    ).reshape((OBS_WINDOW_X, OBS_WINDOW_Y, OBS_WINDOW_Z))

    # Turtle centre in window coords
    cx = OBS_WINDOW_RADIUS_XZ
    cy = OBS_WINDOW_Y_BELOW
    cz = OBS_WINDOW_RADIUS_XZ

    def _block_at(dx: int, dy: int, dz: int) -> int:
        """Return block ID at offset from turtle centre."""
        wx = cx + dx
        wy = cy + dy
        wz = cz + dz
        if (
            0 <= wx < OBS_WINDOW_X
            and 0 <= wy < OBS_WINDOW_Y
            and 0 <= wz < OBS_WINDOW_Z
        ):
            return int(block_ids[wx, wy, wz])
        return BlockType.BEDROCK

    # Movement fuel check
    if req.fuel < 1:
        mask[Action.FORWARD] = False
        mask[Action.UP] = False
        mask[Action.DOWN] = False

    # Movement collision checks
    move_offsets = {
        Action.FORWARD: (int(fv[0]), int(fv[1]), int(fv[2])),
        Action.UP: (0, 1, 0),
        Action.DOWN: (0, -1, 0),
    }
    for action, (dx, dy, dz) in move_offsets.items():
        blk = _block_at(dx, dy, dz)
        if blk != BlockType.AIR:
            mask[action] = False

    # Dig checks
    dig_offsets = {
        Action.DIG: (int(fv[0]), int(fv[1]), int(fv[2])),
        Action.DIG_UP: (0, 1, 0),
        Action.DIG_DOWN: (0, -1, 0),
    }
    for action, (dx, dy, dz) in dig_offsets.items():
        blk = _block_at(dx, dy, dz)
        if blk == BlockType.AIR or blk == BlockType.BEDROCK:
            mask[action] = False

    return mask


# ---------------------------------------------------------------------------
# Application lifespan — load model on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and VecNormalize stats on application startup."""
    global _model, _vec_normalize  # noqa: PLW0603

    config = DeploymentConfig()

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import VecNormalize

        _model = MaskablePPO.load(config.model_path)
        _vec_normalize = VecNormalize.load(
            config.vecnormalize_path,
            venv=None,  # type: ignore[arg-type]
        )
        _vec_normalize.training = False
        _vec_normalize.norm_reward = False
        logger.info("Model loaded from %s", config.model_path)
    except Exception:
        logger.warning(
            "Could not load model — server will return 503 on /act requests.",
            exc_info=True,
        )
        _model = None
        _vec_normalize = None

    yield

    _model = None
    _vec_normalize = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ProspectRL Inference Server", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
    )


@app.post("/act", response_model=ActResponse)
async def act(req: ActRequest) -> ActResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build observation dict
    obs = _build_observation(req)

    # Apply VecNormalize to scalars only (NOT voxels or preference)
    if _vec_normalize is not None:
        obs["scalars"] = _vec_normalize.normalize_obs(
            {"voxels": obs["voxels"], "scalars": obs["scalars"], "pref": obs["pref"]},
        )["scalars"]

    # Compute action mask
    action_mask = _build_action_mask(req)

    # Get action from model
    action_int, _states = _model.predict(obs, action_masks=action_mask, deterministic=True)
    action_int = int(action_int)

    action_enum = Action(action_int)

    # Compute confidence from action probabilities
    confidence = 1.0
    try:
        import torch

        obs_tensor = {
            k: torch.as_tensor(v).unsqueeze(0).float()
            for k, v in obs.items()
        }
        dist = _model.policy.get_distribution(
            _model.policy.extract_features(obs_tensor, _model.policy.features_extractor),
        )
        probs = dist.distribution.probs.detach().cpu().numpy().flatten()
        confidence = float(probs[action_int])
    except Exception:
        pass

    return ActResponse(
        action=action_int,
        action_name=action_enum.name.lower(),
        confidence=confidence,
    )


@app.post("/preference", response_model=PreferenceResponse)
async def preference(req: PreferenceRequest) -> PreferenceResponse:
    target = req.target.lower().removesuffix("_ore")
    if target not in _ORE_NAME_TO_INDEX:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ore target: {req.target!r}. "
            f"Valid targets: {list(_ORE_NAME_TO_INDEX.keys())}",
        )
    pref = [0.0] * NUM_ORE_TYPES
    pref[_ORE_NAME_TO_INDEX[target]] = 1.0
    return PreferenceResponse(preference=pref)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the inference server."""
    import uvicorn

    config = DeploymentConfig()
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
