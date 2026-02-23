"""Package a trained model for deployment.

Copies the SB3 model archive and VecNormalize statistics into a deployment
directory alongside JSON metadata files describing the observation/action
spaces and the block registry.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from prospect_rl.config import (
    BLOCK_NAME_TO_ID,
    NUM_ACTIONS,
    NUM_ORE_TYPES,
    NUM_VOXEL_CHANNELS,
    OBS_WINDOW_X,
    OBS_WINDOW_Y,
    OBS_WINDOW_Z,
    SCALAR_OBS_DIM,
    Action,
)


def _build_config_json() -> dict:
    """Return observation/action space metadata."""
    return {
        "observation": {
            "voxels_shape": [
                NUM_VOXEL_CHANNELS,
                OBS_WINDOW_Y,
                OBS_WINDOW_X,
                OBS_WINDOW_Z,
            ],
            "scalar_dim": SCALAR_OBS_DIM,
            "pref_dim": NUM_ORE_TYPES,
            "components": {
                "voxels_channels": NUM_VOXEL_CHANNELS,
                "position": 3,
                "facing_onehot": 4,
                "fuel": 1,
                "inventory": NUM_ORE_TYPES,
                "world_height_norm": 1,
            },
        },
        "action": {
            "num_actions": NUM_ACTIONS,
            "names": {int(a): a.name.lower() for a in Action},
        },
    }


def _build_block_registry() -> dict:
    """Return bidirectional block name <-> ID mappings."""
    id_to_name = {v: k for k, v in BLOCK_NAME_TO_ID.items()}
    return {
        "name_to_id": BLOCK_NAME_TO_ID,
        "id_to_name": {
            str(k): v for k, v in id_to_name.items()
        },
    }


def export_for_deployment(
    model_path: str,
    vecnormalize_path: str,
    output_dir: str,
) -> None:
    """Package model artifacts for the inference server.

    Parameters
    ----------
    model_path:
        Path to the SB3 model zip file (e.g. ``model.zip``).
    vecnormalize_path:
        Path to the VecNormalize pickle (e.g. ``vecnormalize.pkl``).
    output_dir:
        Destination directory.  Created if it does not exist.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Copy model and normalization stats
    shutil.copy2(model_path, out / "model.zip")
    shutil.copy2(vecnormalize_path, out / "vecnormalize.pkl")

    # Write config.json
    config_data = _build_config_json()
    (out / "config.json").write_text(
        json.dumps(config_data, indent=2),
    )

    # Write block_registry.json
    registry = _build_block_registry()
    (out / "block_registry.json").write_text(
        json.dumps(registry, indent=2),
    )
