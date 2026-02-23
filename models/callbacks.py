"""Training callbacks for checkpointing, metrics logging, and curriculum.

All callbacks extend SB3's ``BaseCallback`` and are designed to work with
``MaskablePPO`` and the ``MinecraftMiningEnv``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class DriveCheckpointCallback(BaseCallback):
    """Save model and VecNormalize stats at regular intervals.

    Maintains a ``latest.json`` manifest and rotates old checkpoints to
    keep at most ``max_kept`` on disk.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_freq: int = 25_000,
        max_kept: int = 3,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._dir = Path(checkpoint_dir)
        self._save_freq = save_freq
        self._max_kept = max_kept
        self._saved: list[Path] = []

    def _init_callback(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq != 0:
            return True

        step = self.num_timesteps
        ckpt_dir = self._dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = ckpt_dir / "model.zip"
        self.model.save(str(model_path))

        # Save VecNormalize
        vec_norm = self.training_env
        vn_path = ckpt_dir / "vecnormalize.pkl"
        vec_norm.save(str(vn_path))

        # Update latest.json
        manifest = {
            "step": step,
            "model_path": str(model_path),
            "vecnormalize_path": str(vn_path),
        }
        (self._dir / "latest.json").write_text(json.dumps(manifest, indent=2))

        self._saved.append(ckpt_dir)

        # Rotate old checkpoints
        while len(self._saved) > self._max_kept:
            old = self._saved.pop(0)
            if old.exists():
                shutil.rmtree(old)

        if self.verbose:
            print(f"[Checkpoint] Saved at step {step} -> {ckpt_dir}")

        return True


class MetricsCallback(BaseCallback):
    """Log per-episode mining metrics to TensorBoard.

    Tracks ore mined counts, total reward, fuel efficiency, and
    episode length.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_ores: dict[int, list[int]] = {}
        self._episode_rewards: dict[int, list[float]] = {}

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            # Track ore mining
            block_mined = info.get("block_mined")
            if block_mined is not None:
                self._episode_ores.setdefault(i, []).append(block_mined)

            # Check for episode end
            if info.get("terminal_observation") is not None or \
               self.locals.get("dones", [False] * (i + 1))[i]:
                ores = self._episode_ores.pop(i, [])
                ore_count = len(ores)

                fuel = info.get("fuel", 0)
                step_count = info.get("step", 1)
                fuel_eff = ore_count / max(step_count, 1)

                self.logger.record("mining/ores_per_episode", ore_count)
                self.logger.record("mining/fuel_remaining", fuel)
                self.logger.record("mining/fuel_efficiency", fuel_eff)
                self.logger.record("mining/episode_steps", step_count)
                self.logger.record(
                    "mining/explored_count",
                    info.get("explored_count", 0),
                )

        return True


class CurriculumCallback(BaseCallback):
    """Placeholder for curriculum stage advancement (Phase 6).

    For now, simply logs the current stage to TensorBoard.
    """

    def __init__(
        self,
        stage_index: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._stage = stage_index

    def _on_step(self) -> bool:
        self.logger.record("curriculum/stage", self._stage)
        return True
