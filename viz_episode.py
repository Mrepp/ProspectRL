"""Watch a trained model control a turtle in the mining environment.

Loads a MaskablePPO model, runs a full episode, and renders the turtle
moving through the 3D world step-by-step with auto-play and step-through
controls.

Usage::

    # Default: download latest checkpoint from project Drive
    python -m prospect_rl.viz_episode

    # Custom Drive folder link
    python -m prospect_rl.viz_episode --drive "https://drive.google.com/drive/folders/..."

    # Local checkpoint directory (reads latest.json)
    python -m prospect_rl.viz_episode -d checkpoints/stage1 --stage 0

    # Explicit model path
    python -m prospect_rl.viz_episode --model checkpoints/final/model.zip --stage 0
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from prospect_rl.config import (
    CURRICULUM_STAGES,
    NUM_ORE_TYPES,
    ORE_TYPES,
    Action,
    BlockType,
)
from prospect_rl.env.turtle import FACING_VECTORS

from viz_world import BLOCK_COLORS, BLOCK_NAMES, ORE_ORDER

# Default Google Drive checkpoint folder (shared link).
DRIVE_CHECKPOINT_URL = (
    "https://drive.google.com/drive/folders/"
    "1Cl14a51SMKYTr2tfm-jTnsy5ne1CHZRh"
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """Snapshot of a single environment step."""

    step: int
    position: tuple[int, int, int]
    facing: int
    action: int
    action_name: str
    fuel: int
    reward: float
    cumulative_reward: float
    block_mined: int | None
    block_mined_pos: tuple[int, int, int] | None
    inventory: dict[int, int]
    explored_count: int


@dataclass
class Trajectory:
    """Complete episode trajectory with metadata."""

    steps: list[StepRecord]
    initial_position: tuple[int, int, int]
    initial_facing: int
    world_blocks: np.ndarray
    world_size: tuple[int, int, int]
    stage_index: int
    stage_name: str
    preference: np.ndarray
    seed: int
    total_reward: float
    total_ores_mined: int


# Ore name → preference index mapping
_ORE_NAME_TO_IDX: dict[str, int] = {
    "coal": 0, "iron": 1, "gold": 2, "diamond": 3,
    "redstone": 4, "emerald": 5, "lapis": 6, "copper": 7,
}

# Action name lookup
_ACTION_NAMES: dict[int, str] = {
    Action.FORWARD: "Forward",
    Action.UP: "Up",
    Action.DOWN: "Down",
    Action.TURN_LEFT: "Turn Left",
    Action.TURN_RIGHT: "Turn Right",
    Action.DIG: "Dig",
    Action.DIG_UP: "Dig Up",
    Action.DIG_DOWN: "Dig Down",
}

# Facing direction → arrow offset for visualization
_FACING_ARROW: dict[int, np.ndarray] = {
    0: np.array([0, 0, 0.6]),   # North (+z)
    1: np.array([0.6, 0, 0]),   # East (+x)
    2: np.array([0, 0, -0.6]),  # South (-z)
    3: np.array([-0.6, 0, 0]),  # West (-x)
}


# ---------------------------------------------------------------------------
# Episode recording
# ---------------------------------------------------------------------------


def _get_dig_target(
    position: np.ndarray, facing: int, action: int,
) -> np.ndarray | None:
    """Compute the dig target position for a dig action."""
    if action == Action.DIG:
        return position + FACING_VECTORS[facing]
    elif action == Action.DIG_UP:
        return position + np.array([0, 1, 0], dtype=np.int32)
    elif action == Action.DIG_DOWN:
        return position + np.array([0, -1, 0], dtype=np.int32)
    return None


def _parse_preference(pref_str: str) -> np.ndarray:
    """Parse a preference string into a weight vector.

    Accepts ore names (``diamond``, ``iron``) for one-hot, or comma-separated
    floats for explicit weights.
    """
    pref_str = pref_str.strip().lower()
    if pref_str in _ORE_NAME_TO_IDX:
        vec = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
        vec[_ORE_NAME_TO_IDX[pref_str]] = 1.0
        return vec

    parts = pref_str.split(",")
    if len(parts) == NUM_ORE_TYPES:
        vec = np.array([float(p) for p in parts], dtype=np.float32)
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    raise ValueError(
        f"Invalid preference: '{pref_str}'. "
        f"Use an ore name ({', '.join(_ORE_NAME_TO_IDX)}) "
        f"or {NUM_ORE_TYPES} comma-separated weights."
    )


def record_episode(
    model_path: str,
    stage_index: int = 0,
    seed: int = 42,
    preference: np.ndarray | None = None,
    vecnormalize_path: str | None = None,
) -> Trajectory:
    """Load a model, run one episode, and return the full trajectory."""
    from prospect_rl.env.mining_env import MinecraftMiningEnv
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path)
    stage = CURRICULUM_STAGES[stage_index]

    env = MinecraftMiningEnv(curriculum_stage=stage_index, seed=seed)
    obs, info = env.reset()

    # Snapshot the world before the agent starts digging
    if hasattr(env._world, "_grid"):
        world_blocks = env._world._grid.copy()
    else:
        world_blocks = env._world._blocks.copy()
    world_size = env._world.shape

    initial_pos = tuple(int(v) for v in env._turtle.position)
    initial_facing = env._turtle.facing

    # Override preference if provided
    if preference is not None:
        obs["pref"] = preference.copy()
    else:
        preference = obs["pref"].copy()

    steps: list[StepRecord] = []
    cumulative_reward = 0.0
    total_ores = 0
    done = False

    while not done:
        action_mask = env.action_masks()
        action, _ = model.predict(
            obs, action_masks=action_mask, deterministic=True,
        )
        action_int = int(action)

        # Record position and facing BEFORE the step
        pre_pos = env._turtle.position.copy()
        pre_facing = env._turtle.facing

        # Determine dig target (if digging)
        dig_target = _get_dig_target(pre_pos, pre_facing, action_int)
        block_mined_pos = None

        obs, reward, terminated, truncated, info = env.step(action_int)

        # Override preference each step
        if preference is not None:
            obs["pref"] = preference.copy()

        cumulative_reward += reward

        # Check if a block was actually mined
        block_mined = info.get("block_mined")
        if block_mined is not None:
            if dig_target is not None:
                block_mined_pos = tuple(int(v) for v in dig_target)
            if int(block_mined) in [int(o) for o in ORE_TYPES]:
                total_ores += 1

        step_pos = tuple(int(v) for v in env._turtle.position)

        steps.append(StepRecord(
            step=len(steps),
            position=step_pos,
            facing=env._turtle.facing,
            action=action_int,
            action_name=_ACTION_NAMES.get(action_int, f"Action({action_int})"),
            fuel=env._turtle.fuel,
            reward=float(reward),
            cumulative_reward=cumulative_reward,
            block_mined=block_mined,
            block_mined_pos=block_mined_pos,
            inventory=dict(env._turtle.inventory),
            explored_count=info.get("explored_count", len(env._explored)),
        ))

        done = terminated or truncated

    return Trajectory(
        steps=steps,
        initial_position=initial_pos,
        initial_facing=initial_facing,
        world_blocks=world_blocks,
        world_size=world_size,
        stage_index=stage_index,
        stage_name=stage.name,
        preference=preference,
        seed=seed,
        total_reward=cumulative_reward,
        total_ores_mined=total_ores,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _voxels_for_block(
    blocks: np.ndarray, block_id: int,
) -> pv.PolyData | None:
    """Create unit-cube mesh for every voxel matching *block_id*."""
    xs, ys, zs = np.where(blocks == block_id)
    if len(xs) == 0:
        return None
    centers = np.column_stack([xs, ys, zs]).astype(float)
    cloud = pv.PolyData(centers)
    return cloud.glyph(
        geom=pv.Cube(x_length=0.9, y_length=0.9, z_length=0.9),
    )


class EpisodeVisualizer:
    """Renders a recorded trajectory in a PyVista 3D window."""

    def __init__(
        self,
        trajectory: Trajectory,
        autoplay: bool = True,
        speed_ms: int = 200,
        show_trail: bool = True,
    ) -> None:
        self._traj = trajectory
        self._autoplay = autoplay
        self._speed_ms = speed_ms
        self._show_trail = show_trail
        self._current_step = -1  # -1 means showing initial state
        self._playing = autoplay
        self._timer_id: int | None = None
        self._pl: pv.Plotter | None = None

        # Track which blocks have been mined (for removal from world render)
        self._mined_positions: set[tuple[int, int, int]] = set()
        # Copy blocks so we can mutate during playback
        self._live_blocks = trajectory.world_blocks.copy()

    def show(self) -> None:
        """Build the plotter and start the visualization."""
        pl = pv.Plotter(title=f"ProspectRL Episode — {self._traj.stage_name}")
        pl.set_background("#1a1a2e")
        self._pl = pl

        # Render the initial world (ores + bedrock, caves semi-transparent)
        self._render_world(pl)

        # Place turtle at initial position
        self._render_turtle(
            pl,
            self._traj.initial_position,
            self._traj.initial_facing,
        )

        # HUD text
        self._render_hud(pl, initial=True)

        # Camera
        size = self._traj.world_size
        center = np.array([size[0] / 2, size[1] / 2, size[2] / 2])
        max_dim = max(size)
        pl.camera.focal_point = center
        pl.camera.position = center + np.array([
            max_dim * 1.5, max_dim * 1.2, max_dim * 1.5,
        ])

        # Legend
        pl.add_legend(bcolor=(0.1, 0.1, 0.2, 0.8), face=None)
        pl.add_axes(xlabel="X", ylabel="Y (height)", zlabel="Z")

        # Key bindings
        pl.add_key_event("space", self._on_space)
        pl.add_key_event("b", self._on_back)
        pl.add_key_event("p", self._on_toggle_play)
        pl.add_key_event("plus", self._on_speed_up)
        pl.add_key_event("minus", self._on_speed_down)
        pl.add_key_event("r", self._on_reset)

        # Auto-play timer
        if self._playing:
            self._start_timer(pl)

        # Instructions overlay
        controls_text = (
            "Controls:\n"
            "  Space  step forward\n"
            "  B      step back\n"
            "  P      toggle play\n"
            "  +/-    speed\n"
            "  R      reset"
        )
        pl.add_text(
            controls_text,
            position="lower_left",
            font_size=8,
            color="#888888",
            name="controls",
        )

        print("Opening episode viewer... (close window to exit)")
        print(f"  Episode: {len(self._traj.steps)} steps, "
              f"reward={self._traj.total_reward:.2f}, "
              f"ores={self._traj.total_ores_mined}")
        pl.show()

    # ---- World rendering --------------------------------------------------

    def _render_world(self, pl: pv.Plotter) -> None:
        """Render ore blocks, caves, and bedrock."""
        blocks = self._traj.world_blocks

        for bt in ORE_ORDER:
            mesh = _voxels_for_block(blocks, bt)
            if mesh is not None:
                name = BLOCK_NAMES[bt]
                color = BLOCK_COLORS[bt]
                count = int(np.sum(blocks == bt))
                pl.add_mesh(
                    mesh, color=color,
                    label=f"{name} ({count:,})",
                    opacity=1.0,
                    name=f"ore_{bt}",
                )

        # Caves
        air_mesh = _voxels_for_block(blocks, BlockType.AIR)
        if air_mesh is not None:
            pl.add_mesh(
                air_mesh,
                color=BLOCK_COLORS[BlockType.AIR],
                label="Caves",
                opacity=0.1,
                name="caves",
            )

        # Bedrock floor
        bed_mesh = _voxels_for_block(blocks, BlockType.BEDROCK)
        if bed_mesh is not None:
            pl.add_mesh(
                bed_mesh,
                color=BLOCK_COLORS[BlockType.BEDROCK],
                label="Bedrock",
                opacity=0.25,
                name="bedrock",
            )

    def _rebuild_ore_layer(self, block_type: int) -> None:
        """Re-render a single ore type after a block is mined."""
        if self._pl is None:
            return
        mesh = _voxels_for_block(self._live_blocks, block_type)
        name = f"ore_{block_type}"
        if mesh is not None:
            self._pl.add_mesh(
                mesh,
                color=BLOCK_COLORS.get(block_type, "#ffffff"),
                opacity=1.0,
                name=name,
            )
        else:
            # All blocks of this type have been mined — remove the actor
            self._pl.remove_actor(name)

    # ---- Turtle rendering -------------------------------------------------

    def _render_turtle(
        self,
        pl: pv.Plotter,
        position: tuple[int, int, int],
        facing: int,
    ) -> None:
        """Draw the turtle body + facing arrow."""
        px, py, pz = position

        # Body: green cube
        body = pv.Cube(
            center=(px, py, pz),
            x_length=0.95, y_length=0.95, z_length=0.95,
        )
        pl.add_mesh(body, color="#00cc44", opacity=0.9, name="turtle_body")

        # Facing arrow: yellow cone
        arrow_offset = _FACING_ARROW[facing]
        arrow_dir = arrow_offset / np.linalg.norm(arrow_offset)
        arrow = pv.Arrow(
            start=np.array([px, py, pz], dtype=float) + arrow_offset * 0.1,
            direction=arrow_dir,
            scale=0.5,
            tip_length=0.4,
            shaft_radius=0.08,
            tip_radius=0.2,
        )
        pl.add_mesh(arrow, color="#ffdd00", opacity=0.95, name="turtle_arrow")

    # ---- Trail rendering --------------------------------------------------

    def _render_trail(self, pl: pv.Plotter, up_to_step: int) -> None:
        """Draw the path the turtle has taken as a cyan polyline."""
        if not self._show_trail or up_to_step < 0:
            pl.remove_actor("trail")
            return

        points = [np.array(self._traj.initial_position, dtype=float)]
        for i in range(up_to_step + 1):
            p = np.array(self._traj.steps[i].position, dtype=float)
            # Only add if position changed
            if not np.array_equal(p, points[-1]):
                points.append(p)

        if len(points) < 2:
            pl.remove_actor("trail")
            return

        pts = np.array(points)
        line = pv.Spline(pts, n_points=len(pts))
        pl.add_mesh(
            line, color="#00dddd", line_width=3,
            opacity=0.7, name="trail",
        )

    # ---- Mine flash effect ------------------------------------------------

    def _render_mine_flash(
        self, pl: pv.Plotter, pos: tuple[int, int, int],
    ) -> None:
        """Flash a wireframe cube at the mined position."""
        px, py, pz = pos
        box = pv.Cube(
            center=(px, py, pz),
            x_length=1.05, y_length=1.05, z_length=1.05,
        )
        pl.add_mesh(
            box, color="#ff4444", style="wireframe",
            line_width=3, opacity=0.9, name="mine_flash",
        )

    # ---- HUD --------------------------------------------------------------

    def _render_hud(
        self, pl: pv.Plotter, initial: bool = False,
    ) -> None:
        """Update the heads-up display text overlay."""
        traj = self._traj
        stage = CURRICULUM_STAGES[traj.stage_index]

        if initial or self._current_step < 0:
            pos = traj.initial_position
            facing = traj.initial_facing
            fuel = stage.max_fuel
            reward = 0.0
            cum_reward = 0.0
            action_name = "—"
            inv_str = "empty"
            explored = 1
            step_num = 0
        else:
            rec = traj.steps[self._current_step]
            pos = rec.position
            facing = rec.facing
            fuel = rec.fuel
            reward = rec.reward
            cum_reward = rec.cumulative_reward
            action_name = rec.action_name
            explored = rec.explored_count
            step_num = rec.step + 1

            # Build inventory string
            inv_parts = []
            for bt, count in sorted(rec.inventory.items()):
                name = BLOCK_NAMES.get(bt, str(bt))
                inv_parts.append(f"{name}:{count}")
            inv_str = ", ".join(inv_parts) if inv_parts else "empty"

        facing_names = {0: "N(+z)", 1: "E(+x)", 2: "S(-z)", 3: "W(-x)"}

        # Preference string
        pref_parts = []
        for i, w in enumerate(traj.preference):
            if w > 0.01:
                ore_names = list(_ORE_NAME_TO_IDX.keys())
                pref_parts.append(f"{ore_names[i]}:{w:.2f}")
        pref_str = ", ".join(pref_parts) if pref_parts else "uniform"

        play_state = "PLAYING" if self._playing else "PAUSED"

        hud = (
            f"Step: {step_num}/{len(traj.steps)}  [{play_state}]\n"
            f"Action: {action_name}\n"
            f"Position: {pos}  Facing: {facing_names.get(facing, '?')}\n"
            f"Fuel: {fuel}\n"
            f"Reward: {reward:+.3f}  Total: {cum_reward:+.3f}\n"
            f"Inventory: {inv_str}\n"
            f"Explored: {explored}\n"
            f"Preference: {pref_str}\n"
            f"Speed: {self._speed_ms}ms/step"
        )
        pl.add_text(
            hud, position="upper_right",
            font_size=9, color="white", name="hud",
        )

        # Title
        title = (
            f"Stage {traj.stage_index + 1}: {traj.stage_name}  "
            f"(seed={traj.seed})"
        )
        pl.add_text(
            title, position="upper_left",
            font_size=11, color="white", name="title",
        )

    # ---- Step navigation --------------------------------------------------

    def _go_to_step(self, target_step: int) -> None:
        """Navigate to a specific step index (-1 = initial state)."""
        if self._pl is None:
            return

        max_step = len(self._traj.steps) - 1
        target_step = max(-1, min(target_step, max_step))

        if target_step == self._current_step:
            return

        # If going backward, we need to rebuild the world state
        if target_step < self._current_step:
            self._rebuild_to_step(target_step)
        else:
            # Going forward: apply mined blocks incrementally
            for i in range(self._current_step + 1, target_step + 1):
                rec = self._traj.steps[i]
                if rec.block_mined is not None and rec.block_mined_pos is not None:
                    bx, by, bz = rec.block_mined_pos
                    if 0 <= bx < self._traj.world_size[0] and \
                       0 <= by < self._traj.world_size[1] and \
                       0 <= bz < self._traj.world_size[2]:
                        self._live_blocks[bx, by, bz] = BlockType.AIR
                        self._mined_positions.add(rec.block_mined_pos)
                        self._rebuild_ore_layer(rec.block_mined)

        self._current_step = target_step

        # Update turtle position
        if target_step < 0:
            pos = self._traj.initial_position
            facing = self._traj.initial_facing
        else:
            rec = self._traj.steps[target_step]
            pos = rec.position
            facing = rec.facing

        self._render_turtle(self._pl, pos, facing)
        self._render_trail(self._pl, target_step)
        self._render_hud(self._pl)

        # Mine flash for current step
        if target_step >= 0:
            rec = self._traj.steps[target_step]
            if rec.block_mined_pos is not None:
                self._render_mine_flash(self._pl, rec.block_mined_pos)
            else:
                self._pl.remove_actor("mine_flash")
        else:
            self._pl.remove_actor("mine_flash")

        self._pl.render()

    def _rebuild_to_step(self, target_step: int) -> None:
        """Reset world state and replay up to target_step."""
        self._live_blocks = self._traj.world_blocks.copy()
        self._mined_positions.clear()

        # Rebuild the full world render
        self._render_world(self._pl)

        # Replay all mining events up to target_step
        for i in range(target_step + 1):
            rec = self._traj.steps[i]
            if rec.block_mined is not None and rec.block_mined_pos is not None:
                bx, by, bz = rec.block_mined_pos
                if 0 <= bx < self._traj.world_size[0] and \
                   0 <= by < self._traj.world_size[1] and \
                   0 <= bz < self._traj.world_size[2]:
                    self._live_blocks[bx, by, bz] = BlockType.AIR
                    self._mined_positions.add(rec.block_mined_pos)

        # Rebuild all ore layers that had any blocks mined
        mined_types = set()
        for i in range(target_step + 1):
            rec = self._traj.steps[i]
            if rec.block_mined is not None:
                mined_types.add(rec.block_mined)
        for bt in mined_types:
            self._rebuild_ore_layer(bt)

    # ---- Timer / playback -------------------------------------------------

    def _start_timer(self, pl: pv.Plotter) -> None:
        """Start the auto-play timer."""
        if self._timer_id is not None:
            return
        self._timer_id = pl.add_timer_event(
            max_steps=0,  # infinite
            duration=self._speed_ms,
            callback=self._on_timer_tick,
        )

    def _stop_timer(self) -> None:
        """Stop the auto-play timer."""
        # PyVista doesn't expose a remove_timer API directly,
        # so we guard via _playing flag instead
        self._timer_id = None

    def _on_timer_tick(self, *args: object) -> None:
        """Timer callback for auto-play."""
        if not self._playing or self._pl is None:
            return
        if self._current_step >= len(self._traj.steps) - 1:
            self._playing = False
            self._render_hud(self._pl)
            self._pl.render()
            return
        self._go_to_step(self._current_step + 1)

    # ---- Key event handlers -----------------------------------------------

    def _on_space(self) -> None:
        """Advance one step."""
        self._playing = False
        self._go_to_step(self._current_step + 1)

    def _on_back(self) -> None:
        """Go back one step."""
        self._playing = False
        self._go_to_step(self._current_step - 1)

    def _on_toggle_play(self) -> None:
        """Toggle auto-play."""
        self._playing = not self._playing
        if self._playing and self._pl is not None:
            # Restart from beginning if at the end
            if self._current_step >= len(self._traj.steps) - 1:
                self._go_to_step(-1)
            self._start_timer(self._pl)
        if self._pl is not None:
            self._render_hud(self._pl)
            self._pl.render()

    def _on_speed_up(self) -> None:
        """Decrease interval (speed up)."""
        self._speed_ms = max(20, self._speed_ms - 50)
        if self._pl is not None:
            self._render_hud(self._pl)
            self._pl.render()

    def _on_speed_down(self) -> None:
        """Increase interval (slow down)."""
        self._speed_ms = min(2000, self._speed_ms + 50)
        if self._pl is not None:
            self._render_hud(self._pl)
            self._pl.render()

    def _on_reset(self) -> None:
        """Reset to initial state."""
        self._playing = False
        self._go_to_step(-1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _download_from_drive(url: str) -> str:
    """Download a checkpoint folder from Google Drive.

    Uses ``gdown`` to pull the shared folder into a local cache
    directory.  Files that already exist are skipped (``resume=True``).

    Returns the path to the local directory containing the downloaded
    checkpoint files.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for --drive. "
            "Install it with: pip install gdown"
        ) from None

    cache_dir = Path(tempfile.gettempdir()) / "prospect_rl_drive"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoint from Drive...")
    paths = gdown.download_folder(
        url=url,
        output=str(cache_dir),
        quiet=False,
        remaining_ok=True,
        resume=True,
    )
    if not paths:
        raise RuntimeError(
            f"gdown returned no files for {url}. "
            "Check that the folder is shared with "
            "'Anyone with the link'."
        )

    # gdown creates a subfolder named after the Drive folder.
    # Find it — it's the newest directory inside cache_dir.
    subdirs = sorted(
        [p for p in cache_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if subdirs:
        return str(subdirs[0])
    return str(cache_dir)


def _resolve_checkpoint(
    checkpoint_dir: str,
    stage_index: int | None = None,
) -> tuple[str, str | None, int | None]:
    """Resolve model and vecnormalize paths from a checkpoint dir.

    Returns ``(model_path, vecnormalize_path, detected_stage)``.
    ``detected_stage`` is inferred from folder names like ``stage1``
    when *stage_index* is ``None``.

    Supports:
      - Directory with ``latest.json``
      - Step directory containing ``model.zip`` directly
      - Parent directory with stage subdirs (e.g. ``stage1/``)
    """
    d = Path(checkpoint_dir)

    # Direct hit — latest.json in this directory
    manifest = d / "latest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        model_p = Path(data["model_path"])
        vn_p = Path(data["vecnormalize_path"]) if data.get("vecnormalize_path") else None

        # Manifest may contain absolute Colab paths that don't
        # exist locally.  Fall back to the same relative structure
        # under the checkpoint directory.
        if not model_p.exists():
            model_p = d / model_p.parent.name / model_p.name
        if vn_p and not vn_p.exists():
            vn_p = d / vn_p.parent.name / vn_p.name

        return (
            str(model_p),
            str(vn_p) if vn_p else None,
            stage_index,
        )

    # Direct hit — model.zip in this directory
    model = d / "model.zip"
    if model.exists():
        vn = d / "vecnormalize.pkl"
        return (
            str(model),
            str(vn) if vn.exists() else None,
            stage_index,
        )

    # Search subdirs for stage folders with latest.json.
    # Pick the requested stage, or the highest-numbered one.
    candidates: list[tuple[int, Path]] = []
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        m = sub / "latest.json"
        if not m.exists():
            m2 = sub / "final" / "model.zip"
            if not m2.exists():
                continue
        # Try to extract stage number from folder name
        name = sub.name.lower()
        stage_num: int | None = None
        for i, cs in enumerate(CURRICULUM_STAGES):
            if name == cs.name or name == f"stage{i + 1}":
                stage_num = i
                break
        candidates.append((stage_num if stage_num is not None else -1, sub))

    if candidates:
        if stage_index is not None:
            # Pick the matching stage
            for sn, sub in candidates:
                if sn == stage_index:
                    return _resolve_checkpoint(
                        str(sub), stage_index,
                    )
        # Default: pick highest stage number
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_stage, best_sub = candidates[0]
        detected = best_stage if best_stage >= 0 else None
        return _resolve_checkpoint(
            str(best_sub), detected,
        )

    raise FileNotFoundError(
        f"No latest.json or model.zip found in {d}"
    )


def main() -> None:
    n = len(CURRICULUM_STAGES)
    stage_help = [
        f"  {i}: {s.name} "
        f"({s.world_size[0]}x{s.world_size[1]}x{s.world_size[2]})"
        for i, s in enumerate(CURRICULUM_STAGES)
    ]

    parser = argparse.ArgumentParser(
        description="Watch a trained model play an episode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Curriculum stages:\n" + "\n".join(stage_help),
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Path to model.zip (MaskablePPO)",
    )
    parser.add_argument(
        "--checkpoint-dir", "-d", type=str, default=None,
        help=(
            "Checkpoint directory (reads latest.json "
            "or model.zip inside)"
        ),
    )
    parser.add_argument(
        "--drive", type=str, default=None, nargs="?",
        const=DRIVE_CHECKPOINT_URL,
        help=(
            "Google Drive shareable folder URL. "
            "If used without a URL, defaults to the "
            "project checkpoint folder."
        ),
    )
    parser.add_argument(
        "--vecnormalize", type=str, default=None,
        help="Path to vecnormalize.pkl (optional)",
    )
    parser.add_argument(
        "--stage", type=int, default=None, choices=range(n),
        help=(
            f"Curriculum stage 0-{n-1}. "
            "Auto-detected from checkpoint folder when omitted."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Environment seed (default: 42)",
    )
    parser.add_argument(
        "--preference", type=str, default=None,
        help=(
            "Ore name (e.g. 'diamond') or "
            "comma-separated weights"
        ),
    )
    parser.add_argument(
        "--speed", type=int, default=200,
        help="Auto-play speed in ms/step (default: 200)",
    )
    parser.add_argument(
        "--no-autoplay", action="store_true",
        help="Start paused (step-through mode)",
    )
    parser.add_argument(
        "--no-trail", action="store_true",
        help="Hide the turtle's path trail",
    )

    args = parser.parse_args()

    # Resolve model source. Default to Drive when nothing given.
    sources = [args.drive, args.checkpoint_dir, args.model]
    n_sources = sum(s is not None for s in sources)
    if n_sources > 1:
        parser.error(
            "Provide at most one of "
            "--drive, --checkpoint-dir, or --model"
        )
    if n_sources == 0:
        # No source specified — default to Drive
        args.drive = DRIVE_CHECKPOINT_URL

    model_path = args.model
    vn_path = args.vecnormalize
    stage_index = args.stage

    if args.drive:
        local_dir = _download_from_drive(args.drive)
        model_path, resolved_vn, detected_stage = (
            _resolve_checkpoint(local_dir, stage_index)
        )
        if vn_path is None:
            vn_path = resolved_vn
        if stage_index is None and detected_stage is not None:
            stage_index = detected_stage
    elif args.checkpoint_dir:
        model_path, resolved_vn, detected_stage = (
            _resolve_checkpoint(
                args.checkpoint_dir, stage_index,
            )
        )
        if vn_path is None:
            vn_path = resolved_vn
        if stage_index is None and detected_stage is not None:
            stage_index = detected_stage
    else:
        # --model: auto-detect vecnormalize next to model
        if vn_path is None:
            sibling = Path(args.model).parent / "vecnormalize.pkl"
            if sibling.exists():
                vn_path = str(sibling)

    if stage_index is None:
        stage_index = 0
        print("No stage detected, defaulting to stage 0")

    # Parse preference
    pref = None
    if args.preference:
        pref = _parse_preference(args.preference)
        ore_names = list(_ORE_NAME_TO_IDX.keys())
        pref_desc = ", ".join(
            f"{ore_names[i]}={pref[i]:.2f}"
            for i in range(len(pref))
            if pref[i] > 0.01
        )
        print(f"Preference override: {pref_desc}")

    # Record episode
    print(f"Loading model from {model_path}...")
    if vn_path:
        print(f"Using vecnormalize from {vn_path}")
    print(
        f"Running episode "
        f"(stage={stage_index}, seed={args.seed})..."
    )

    trajectory = record_episode(
        model_path=model_path,
        stage_index=stage_index,
        seed=args.seed,
        preference=pref,
        vecnormalize_path=vn_path,
    )

    print(f"Episode complete: {len(trajectory.steps)} steps, "
          f"reward={trajectory.total_reward:.2f}, "
          f"ores={trajectory.total_ores_mined}")

    # Visualize
    viz = EpisodeVisualizer(
        trajectory=trajectory,
        autoplay=not args.no_autoplay,
        speed_ms=args.speed,
        show_trail=not args.no_trail,
    )
    viz.show()


if __name__ == "__main__":
    main()
