"""Multi-turtle episode visualizer.

Spawns multiple turtles with different ore preferences on the same world
and replays their trajectories simultaneously.  Trails persist across
world resets so you can see how each turtle transits to different
Y-levels depending on its target ore.

Performance: the model is loaded once and shared across all turtles.
Turtles are recorded in parallel threads at startup.

Usage::

    # One turtle per ore type (default)
    python viz_multi.py -m checkpoints/model.zip

    # Specific ores with counts
    python viz_multi.py -m model.zip --turtles diamond:3,iron:2

    # 2 of each ore type
    python viz_multi.py -m model.zip --count 2

Controls (inside the 3D window)::

    Space       step forward (all turtles)
    B           step backward
    P           toggle auto-play
    +/-         speed up / slow down
    R           reset playback to step 0
    W           reset world (random seed, re-records all turtles)
    1-9         select turtle
    D           delete selected turtle
    H           toggle trail visibility for selected turtle
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

try:
    import vtk
except ImportError:
    vtk = None  # type: ignore[assignment]

from prospect_rl.config import (
    CURRICULUM_STAGES,
    NUM_ORE_TYPES,
    ORE_TYPES,
    Action,
    BlockType,
)
from prospect_rl.env.turtle import FACING_VECTORS

from viz_episode import (
    DRIVE_CHECKPOINT_URL,
    StepRecord,
    Trajectory,
    _ACTION_NAMES,
    _FACING_ARROW,
    _ORE_NAME_TO_IDX,
    _download_from_drive,
    _parse_preference,
    _resolve_checkpoint,
    _voxels_for_block,
)
from viz_world import BLOCK_COLORS, BLOCK_NAMES, ORE_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TURTLE_COLORS = [
    "#00cc44",  # green
    "#ff6644",  # orange-red
    "#4488ff",  # blue
    "#ffcc00",  # yellow
    "#cc44ff",  # purple
    "#00cccc",  # teal
    "#ff44aa",  # pink
    "#88cc00",  # lime
    "#ff8800",  # orange
    "#4444ff",  # indigo
    "#cc8844",  # brown
    "#44ffcc",  # mint
]

_ORE_NAMES = list(_ORE_NAME_TO_IDX.keys())

_ACTION_SHORT: dict[int, str] = {
    Action.FORWARD: "Fwd",
    Action.UP: "Up",
    Action.DOWN: "Down",
    Action.TURN_LEFT: "TnL",
    Action.TURN_RIGHT: "TnR",
    Action.DIG: "Dig",
    Action.DIG_UP: "DgU",
    Action.DIG_DOWN: "DgD",
}


# ---------------------------------------------------------------------------
# Fast episode recording (model loaded once, shared across calls)
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


class _ModelCache:
    """Loads and caches the MaskablePPO model so it's only loaded once."""

    def __init__(self) -> None:
        self._model = None
        self._model_path: str | None = None
        self._lock = threading.Lock()

    def get(self, model_path: str):
        with self._lock:
            if self._model is None or self._model_path != model_path:
                from sb3_contrib import MaskablePPO
                print("  Loading model (one-time)...")
                self._model = MaskablePPO.load(model_path)
                self._model_path = model_path
            return self._model


_model_cache = _ModelCache()


def record_episode_fast(
    model_path: str,
    stage_index: int = 0,
    seed: int = 42,
    preference: np.ndarray | None = None,
    vecnormalize_path: str | None = None,
) -> Trajectory:
    """Record an episode reusing the cached model."""
    from prospect_rl.env.mining_env import MinecraftMiningEnv

    model = _model_cache.get(model_path)
    stage = CURRICULUM_STAGES[stage_index]

    env = MinecraftMiningEnv(curriculum_stage=stage_index, seed=seed)
    obs, info = env.reset()

    if hasattr(env._world, "_grid"):
        world_blocks = env._world._grid.copy()
    else:
        world_blocks = env._world._blocks.copy()
    world_size = env._world.shape

    initial_pos = tuple(int(v) for v in env._turtle.position)
    initial_facing = env._turtle.facing

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

        pre_pos = env._turtle.position.copy()
        pre_facing = env._turtle.facing
        dig_target = _get_dig_target(pre_pos, pre_facing, action_int)
        block_mined_pos = None

        obs, reward, terminated, truncated, info = env.step(action_int)

        if preference is not None:
            obs["pref"] = preference.copy()

        cumulative_reward += reward

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
            action_name=_ACTION_NAMES.get(
                action_int, f"Action({action_int})",
            ),
            fuel=env._turtle.fuel,
            reward=float(reward),
            cumulative_reward=cumulative_reward,
            block_mined=block_mined,
            block_mined_pos=block_mined_pos,
            inventory=dict(env._turtle.inventory),
            explored_count=info.get(
                "explored_count", len(env._explored),
            ),
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
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrajectorySegment:
    """One recording under a single preference vector."""

    trajectory: Trajectory
    preference_label: str
    trail_opacity: float = 0.7


@dataclass
class TurtleSession:
    """One turtle's lifecycle."""

    turtle_id: int
    color: str
    preference: np.ndarray
    segments: list[TrajectorySegment] = field(default_factory=list)
    active_segment_idx: int = 0
    visible: bool = True


# ---------------------------------------------------------------------------
# Turtle spec parsing
# ---------------------------------------------------------------------------


def _parse_turtle_specs(
    turtles_arg: str | None, count: int,
) -> list[tuple[str, np.ndarray]]:
    """Parse CLI args into a list of (label, preference) tuples.

    Returns one entry per turtle to spawn.

    Examples::

        None, count=1  -> one of each ore type (8 turtles)
        None, count=2  -> two of each ore type (16 turtles)
        "diamond,iron", count=1 -> diamond x1, iron x1
        "diamond:3,iron:2", count=1 -> diamond x3, iron x2
    """
    specs: list[tuple[str, np.ndarray]] = []

    if turtles_arg is None:
        # Default: one (or count) of each ore type
        for ore_name in _ORE_NAMES:
            pref = np.zeros(NUM_ORE_TYPES, dtype=np.float32)
            pref[_ORE_NAME_TO_IDX[ore_name]] = 1.0
            label = ore_name.capitalize()
            for _ in range(count):
                specs.append((label, pref.copy()))
        return specs

    for token in turtles_arg.split(","):
        token = token.strip()
        if ":" in token:
            parts = token.rsplit(":", 1)
            ore_part = parts[0].strip().lower()
            try:
                n = int(parts[1])
            except ValueError:
                # Might be a comma-weight string, try parsing whole thing
                pref = _parse_preference(token)
                label = _preference_label_static(pref)
                specs.append((label, pref))
                continue
        else:
            ore_part = token.lower()
            n = count

        pref = _parse_preference(ore_part)
        label = _preference_label_static(pref)
        for _ in range(n):
            specs.append((label, pref.copy()))

    return specs


def _preference_label_static(pref: np.ndarray) -> str:
    """Derive a short label from a preference vector."""
    idx = int(np.argmax(pref))
    if pref[idx] > 0.9:
        return _ORE_NAMES[idx].capitalize()
    top2 = np.argsort(pref)[-2:][::-1]
    parts = [f"{_ORE_NAMES[i]}:{pref[i]:.1f}" for i in top2]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


class MultiTurtleVisualizer:
    """Renders multiple turtles replaying simultaneously on a shared world."""

    def __init__(
        self,
        model_path: str,
        stage_index: int = 0,
        seed: int = 42,
        speed_ms: int = 200,
        vecnormalize_path: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._stage_index = stage_index
        self._seed = seed
        self._speed_ms = speed_ms
        self._vn_path = vecnormalize_path

        # Shared world (set on first recording)
        self._world_blocks: np.ndarray | None = None
        self._world_size: tuple[int, int, int] | None = None

        # Turtle sessions
        self._turtles: list[TurtleSession] = []
        self._next_turtle_id: int = 0
        self._selected_idx: int | None = None
        self._hovered_idx: int | None = None

        # Original specs for world reset (re-records same turtles)
        self._turtle_specs: list[tuple[str, np.ndarray]] = []

        # Playback
        self._current_step: int = -1
        self._max_step: int = 0
        self._playing: bool = False
        self._timer_id: int | None = None
        self._pl: pv.Plotter | None = None

        # Actor -> turtle reverse map for hover picking
        self._actor_to_turtle: dict[int, int] = {}

        # Warm up model cache
        _model_cache.get(model_path)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_all(
        self, specs: list[tuple[str, np.ndarray]],
    ) -> None:
        """Record trajectories for all turtle specs in parallel.

        The first turtle is recorded synchronously to capture the world
        snapshot.  Remaining turtles are recorded in parallel threads.
        """
        self._turtle_specs = [
            (label, pref.copy()) for label, pref in specs
        ]
        self._turtles.clear()
        self._next_turtle_id = 0
        self._max_step = 0
        self._current_step = -1

        if not specs:
            return

        # First turtle sync (captures world)
        first_label, first_pref = specs[0]
        print(f"  Recording '{first_label}' (seed={self._seed})...")
        first_traj = record_episode_fast(
            self._model_path, self._stage_index,
            self._seed, first_pref, self._vn_path,
        )
        self._world_blocks = first_traj.world_blocks.copy()
        self._world_size = first_traj.world_size
        self._add_turtle(first_label, first_pref, first_traj)
        print(f"    {first_label}: {len(first_traj.steps)} steps")

        # Remaining turtles in parallel
        remaining = specs[1:]
        if not remaining:
            return

        n_workers = min(8, len(remaining))
        print(f"  Recording {len(remaining)} more turtle(s) "
              f"({n_workers} threads)...")

        def _record(spec):
            label, pref = spec
            return spec, record_episode_fast(
                self._model_path, self._stage_index,
                self._seed, pref, self._vn_path,
            )

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_record, s) for s in remaining]
            for future in as_completed(futures):
                (label, pref), traj = future.result()
                self._add_turtle(label, pref, traj)
                print(f"    {label}: {len(traj.steps)} steps")

        if self._turtles:
            self._selected_idx = 0

    def _add_turtle(
        self, label: str, pref: np.ndarray, traj: Trajectory,
    ) -> TurtleSession:
        """Add a fully-recorded turtle to the session list."""
        tid = self._next_turtle_id
        self._next_turtle_id += 1
        color = TURTLE_COLORS[tid % len(TURTLE_COLORS)]

        seg = TrajectorySegment(
            trajectory=traj,
            preference_label=label,
        )
        ts = TurtleSession(
            turtle_id=tid,
            color=color,
            preference=pref.copy(),
            segments=[seg],
        )
        self._turtles.append(ts)
        self._max_step = max(
            self._max_step, len(traj.steps) - 1,
        )
        return ts

    # ------------------------------------------------------------------
    # World rendering
    # ------------------------------------------------------------------

    def _render_world(self, pl: pv.Plotter) -> None:
        """Render ore blocks, caves, and bedrock (static)."""
        if self._world_blocks is None:
            return
        blocks = self._world_blocks
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

        air_mesh = _voxels_for_block(blocks, BlockType.AIR)
        if air_mesh is not None:
            pl.add_mesh(
                air_mesh,
                color=BLOCK_COLORS[BlockType.AIR],
                label="Caves",
                opacity=0.1,
                name="caves",
            )

        bed_mesh = _voxels_for_block(blocks, BlockType.BEDROCK)
        if bed_mesh is not None:
            pl.add_mesh(
                bed_mesh,
                color=BLOCK_COLORS[BlockType.BEDROCK],
                label="Bedrock",
                opacity=0.25,
                name="bedrock",
            )

    # ------------------------------------------------------------------
    # Turtle rendering
    # ------------------------------------------------------------------

    def _render_turtle_at_step(
        self, ts: TurtleSession, step: int,
    ) -> None:
        """Draw one turtle's body + arrow at the given step."""
        if self._pl is None:
            return
        pl = self._pl
        seg = ts.segments[ts.active_segment_idx]
        traj = seg.trajectory

        if step < 0:
            pos = traj.initial_position
            facing = traj.initial_facing
        else:
            clamped = min(step, len(traj.steps) - 1)
            rec = traj.steps[clamped]
            pos = rec.position
            facing = rec.facing

        px, py, pz = pos

        # Body
        body_name = f"turtle_body_{ts.turtle_id}"
        body = pv.Cube(
            center=(px, py, pz),
            x_length=0.95, y_length=0.95, z_length=0.95,
        )
        pl.add_mesh(
            body, color=ts.color, opacity=0.9, name=body_name,
        )
        if body_name in pl.renderer.actors:
            actor = pl.renderer.actors[body_name]
            self._actor_to_turtle[id(actor)] = ts.turtle_id

        # Arrow
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
        pl.add_mesh(
            arrow, color=ts.color, opacity=0.95,
            name=f"turtle_arrow_{ts.turtle_id}",
        )

        # Selection / hover highlight
        highlight_name = f"turtle_highlight_{ts.turtle_id}"
        tidx = next(
            (i for i, t in enumerate(self._turtles)
             if t.turtle_id == ts.turtle_id),
            None,
        )
        if tidx is not None and (
            tidx == self._selected_idx or tidx == self._hovered_idx
        ):
            halo = pv.Cube(
                center=(px, py, pz),
                x_length=1.3, y_length=1.3, z_length=1.3,
            )
            pl.add_mesh(
                halo, color="#ffffff", style="wireframe",
                line_width=2, opacity=0.8, name=highlight_name,
            )
        else:
            pl.remove_actor(highlight_name)

        # Mine flash
        flash_name = f"mine_flash_{ts.turtle_id}"
        if step >= 0:
            clamped = min(step, len(traj.steps) - 1)
            rec = traj.steps[clamped]
            if rec.block_mined_pos is not None:
                fx, fy, fz = rec.block_mined_pos
                box = pv.Cube(
                    center=(fx, fy, fz),
                    x_length=1.05, y_length=1.05, z_length=1.05,
                )
                pl.add_mesh(
                    box, color=ts.color, style="wireframe",
                    line_width=3, opacity=0.9, name=flash_name,
                )
            else:
                pl.remove_actor(flash_name)
        else:
            pl.remove_actor(flash_name)

    # ------------------------------------------------------------------
    # Trail rendering
    # ------------------------------------------------------------------

    def _render_trails(self, ts: TurtleSession, up_to_step: int) -> None:
        """Render all trail segments for one turtle."""
        if self._pl is None:
            return
        pl = self._pl

        for seg_idx, seg in enumerate(ts.segments):
            trail_name = f"trail_{ts.turtle_id}_seg_{seg_idx}"

            if not ts.visible:
                pl.remove_actor(trail_name)
                continue

            traj = seg.trajectory
            is_active = seg_idx == ts.active_segment_idx

            if is_active:
                max_s = min(up_to_step, len(traj.steps) - 1)
                if max_s < 0:
                    pl.remove_actor(trail_name)
                    continue
                points = [np.array(traj.initial_position, dtype=float)]
                for i in range(max_s + 1):
                    p = np.array(traj.steps[i].position, dtype=float)
                    if not np.array_equal(p, points[-1]):
                        points.append(p)
            else:
                points = [np.array(traj.initial_position, dtype=float)]
                for rec in traj.steps:
                    p = np.array(rec.position, dtype=float)
                    if not np.array_equal(p, points[-1]):
                        points.append(p)

            if len(points) < 2:
                pl.remove_actor(trail_name)
                continue

            pts = np.array(points)
            line = pv.Spline(pts, n_points=len(pts))
            pl.add_mesh(
                line, color=ts.color,
                line_width=3 if is_active else 2,
                opacity=seg.trail_opacity,
                name=trail_name,
            )

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def _render_side_panel(self) -> None:
        """Left-side text overlay listing all turtles."""
        if self._pl is None:
            return

        lines = ["=== TURTLES ===\n"]
        for i, ts in enumerate(self._turtles):
            if not ts.visible:
                continue

            seg = ts.segments[ts.active_segment_idx]
            traj = seg.trajectory

            step = min(self._current_step, len(traj.steps) - 1)

            # Count target ores mined up to current step
            target_ore_bt = int(ORE_TYPES[int(np.argmax(ts.preference))])
            target_mined = 0
            if step >= 0:
                for s in range(step + 1):
                    if traj.steps[s].block_mined is not None and int(traj.steps[s].block_mined) == target_ore_bt:
                        target_mined += 1

            if step < 0:
                reward_str = "+0.000"
                action_str = "---"
                pos_str = (
                    f"({traj.initial_position[0]},"
                    f"{traj.initial_position[1]},"
                    f"{traj.initial_position[2]})"
                )
            else:
                rec = traj.steps[step]
                reward_str = f"{rec.cumulative_reward:+.1f}"
                action_str = _ACTION_SHORT.get(rec.action, "?")
                pos_str = (
                    f"({rec.position[0]},"
                    f"{rec.position[1]},"
                    f"{rec.position[2]})"
                )

            marker = " <<" if i == self._selected_idx else ""
            hover = " *" if i == self._hovered_idx else ""
            lines.append(
                f"[{i + 1}] {seg.preference_label:<10s} "
                f"O:{target_mined:<3d} R:{reward_str}  {action_str:<4s} "
                f"{pos_str}{marker}{hover}"
            )

        panel_text = "\n".join(lines)
        self._pl.add_text(
            panel_text,
            position="upper_left",
            font_size=9,
            color="white",
            name="side_panel",
        )

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _render_hud(self) -> None:
        """Upper-right info panel for the selected/hovered turtle."""
        if self._pl is None:
            return

        show_idx = (
            self._hovered_idx
            if self._hovered_idx is not None
            else self._selected_idx
        )

        if show_idx is None or show_idx >= len(self._turtles):
            play_state = "PLAYING" if self._playing else "PAUSED"
            hud = (
                f"Step: {max(0, self._current_step + 1)}"
                f"/{self._max_step + 1}  [{play_state}]\n"
                f"Turtles: {len(self._turtles)}\n"
                f"Speed: {self._speed_ms}ms/step\n"
                f"Seed: {self._seed}\n\n"
                f"Press 1-9 to select turtle\n"
                f"Press W to reset world"
            )
            self._pl.add_text(
                hud, position="upper_right",
                font_size=9, color="white", name="hud",
            )
            return

        ts = self._turtles[show_idx]
        seg = ts.segments[ts.active_segment_idx]
        traj = seg.trajectory

        step = min(self._current_step, len(traj.steps) - 1)
        facing_names = {0: "N(+z)", 1: "E(+x)", 2: "S(-z)", 3: "W(-x)"}

        if step < 0:
            pos = traj.initial_position
            facing = traj.initial_facing
            stage_cfg = CURRICULUM_STAGES[traj.stage_index]
            fuel_str = (
                "inf" if stage_cfg.infinite_fuel
                else str(stage_cfg.max_fuel)
            )
            reward = 0.0
            cum_reward = 0.0
            action_name = "---"
            inv_str = "empty"
            explored = 1
        else:
            rec = traj.steps[step]
            pos = rec.position
            facing = rec.facing
            fuel_str = str(rec.fuel)
            reward = rec.reward
            cum_reward = rec.cumulative_reward
            action_name = rec.action_name
            explored = rec.explored_count
            inv_parts = []
            for bt, count in sorted(rec.inventory.items()):
                name = BLOCK_NAMES.get(bt, str(bt))
                inv_parts.append(f"{name}:{count}")
            inv_str = ", ".join(inv_parts) if inv_parts else "empty"

        pref_parts = []
        for i, w in enumerate(traj.preference):
            if w > 0.01:
                pref_parts.append(f"{_ORE_NAMES[i]}:{w:.2f}")
        pref_str = ", ".join(pref_parts)

        play_state = "PLAYING" if self._playing else "PAUSED"

        hud = (
            f"Step: {max(0, self._current_step + 1)}"
            f"/{self._max_step + 1}  [{play_state}]\n"
            f"\n"
            f"Turtle {show_idx + 1}: {seg.preference_label}\n"
            f"Preference: {pref_str}\n"
            f"Action: {action_name}\n"
            f"Position: {pos}  Facing: {facing_names.get(facing, '?')}\n"
            f"Fuel: {fuel_str}\n"
            f"Reward: {reward:+.3f}  Total: {cum_reward:+.3f}\n"
            f"Inventory: {inv_str}\n"
            f"Explored: {explored}\n"
            f"Speed: {self._speed_ms}ms/step"
        )
        self._pl.add_text(
            hud, position="upper_right",
            font_size=9, color="white", name="hud",
        )

    # ------------------------------------------------------------------
    # Hover tooltip
    # ------------------------------------------------------------------

    def _render_hover_tooltip(self) -> None:
        """Show floating label near hovered turtle in 3D."""
        if self._pl is None:
            return

        if (self._hovered_idx is None
                or self._hovered_idx >= len(self._turtles)):
            self._pl.remove_actor("hover_tooltip")
            return

        ts = self._turtles[self._hovered_idx]
        seg = ts.segments[ts.active_segment_idx]
        traj = seg.trajectory

        step = min(self._current_step, len(traj.steps) - 1)
        if step < 0:
            pos = traj.initial_position
        else:
            pos = traj.steps[min(step, len(traj.steps) - 1)].position

        label = f"T{self._hovered_idx + 1}: {seg.preference_label}"
        point = np.array([[pos[0], pos[1] + 1.5, pos[2]]], dtype=float)
        cloud = pv.PolyData(point)
        cloud["labels"] = [label]
        self._pl.add_point_labels(
            cloud, "labels",
            font_size=14,
            text_color=ts.color,
            shape_color="black",
            shape_opacity=0.7,
            name="hover_tooltip",
        )

    # ------------------------------------------------------------------
    # Step navigation
    # ------------------------------------------------------------------

    def _go_to_step(self, target_step: int) -> None:
        """Navigate all turtles to the given step."""
        if self._pl is None or not self._turtles:
            return

        target_step = max(-1, min(target_step, self._max_step))
        self._current_step = target_step

        for ts in self._turtles:
            if ts.visible:
                self._render_turtle_at_step(ts, target_step)
                self._render_trails(ts, target_step)

        self._render_side_panel()
        self._render_hud()
        self._render_hover_tooltip()
        self._pl.render()

    # ------------------------------------------------------------------
    # Timer / playback
    # ------------------------------------------------------------------

    def _start_timer(self, pl: pv.Plotter) -> None:
        if self._timer_id is not None:
            return
        self._timer_id = pl.add_timer_event(
            max_steps=0,
            duration=self._speed_ms,
            callback=self._on_timer_tick,
        )

    def _on_timer_tick(self, *args: object) -> None:
        if not self._playing or self._pl is None:
            return
        if self._current_step >= self._max_step:
            self._playing = False
            self._render_hud()
            self._pl.render()
            return
        self._go_to_step(self._current_step + 1)

    # ------------------------------------------------------------------
    # Key event handlers
    # ------------------------------------------------------------------

    def _on_space(self) -> None:
        self._playing = False
        self._go_to_step(self._current_step + 1)

    def _on_back(self) -> None:
        self._playing = False
        self._go_to_step(self._current_step - 1)

    def _on_toggle_play(self) -> None:
        self._playing = not self._playing
        if self._playing and self._pl is not None:
            if self._current_step >= self._max_step:
                self._go_to_step(-1)
            self._start_timer(self._pl)
        if self._pl is not None:
            self._render_hud()
            self._pl.render()

    def _on_speed_up(self) -> None:
        self._speed_ms = max(20, self._speed_ms - 50)
        if self._pl is not None:
            self._render_hud()
            self._pl.render()

    def _on_speed_down(self) -> None:
        self._speed_ms = min(2000, self._speed_ms + 50)
        if self._pl is not None:
            self._render_hud()
            self._pl.render()

    def _on_reset(self) -> None:
        self._playing = False
        self._go_to_step(-1)

    def _make_select_handler(self, idx: int):
        """Create a key handler that selects turtle at index *idx*."""
        def handler() -> None:
            if idx < len(self._turtles):
                self._selected_idx = idx
                self._go_to_step(self._current_step)
        return handler

    def _on_delete(self) -> None:
        """Delete the selected turtle."""
        if self._pl is None or self._selected_idx is None:
            return
        if self._selected_idx >= len(self._turtles):
            return

        ts = self._turtles[self._selected_idx]
        for seg_idx in range(len(ts.segments)):
            self._pl.remove_actor(f"trail_{ts.turtle_id}_seg_{seg_idx}")
        self._pl.remove_actor(f"turtle_body_{ts.turtle_id}")
        self._pl.remove_actor(f"turtle_arrow_{ts.turtle_id}")
        self._pl.remove_actor(f"turtle_highlight_{ts.turtle_id}")
        self._pl.remove_actor(f"mine_flash_{ts.turtle_id}")

        self._turtles.pop(self._selected_idx)
        if self._turtles:
            self._selected_idx = min(
                self._selected_idx, len(self._turtles) - 1,
            )
        else:
            self._selected_idx = None

        if self._turtles:
            self._max_step = max(
                len(t.segments[t.active_segment_idx].trajectory.steps) - 1
                for t in self._turtles
            )
        else:
            self._max_step = 0

        self._render_side_panel()
        self._render_hud()
        self._pl.render()

    def _on_toggle_trail(self) -> None:
        """Toggle trail visibility for selected turtle."""
        if (self._selected_idx is None
                or self._selected_idx >= len(self._turtles)):
            return
        ts = self._turtles[self._selected_idx]
        ts.visible = not ts.visible

        if not ts.visible and self._pl:
            for seg_idx in range(len(ts.segments)):
                self._pl.remove_actor(
                    f"trail_{ts.turtle_id}_seg_{seg_idx}"
                )
        self._go_to_step(self._current_step)

    def _on_world_reset(self) -> None:
        """Reset world with a random seed and re-record all turtles."""
        self._playing = False
        new_seed = np.random.randint(0, 100000)
        print(f"  Resetting world (seed={new_seed})...")

        # Remove all turtle actors
        if self._pl:
            for ts in self._turtles:
                for seg_idx in range(len(ts.segments)):
                    self._pl.remove_actor(
                        f"trail_{ts.turtle_id}_seg_{seg_idx}"
                    )
                self._pl.remove_actor(f"turtle_body_{ts.turtle_id}")
                self._pl.remove_actor(f"turtle_arrow_{ts.turtle_id}")
                self._pl.remove_actor(f"turtle_highlight_{ts.turtle_id}")
                self._pl.remove_actor(f"mine_flash_{ts.turtle_id}")

        self._seed = new_seed
        self._world_blocks = None
        self._world_size = None
        self._actor_to_turtle.clear()

        # Re-record all turtles with same specs
        self.record_all(self._turtle_specs)

        # Re-render world and turtles
        if self._pl:
            self._render_world(self._pl)
            self._go_to_step(-1)

        print(f"  World reset complete (seed={new_seed}, "
              f"{len(self._turtles)} turtles).")

    # ------------------------------------------------------------------
    # Hover detection
    # ------------------------------------------------------------------

    def _setup_hover(self, pl: pv.Plotter) -> None:
        """Set up mouse-move hover detection via VTK interactor."""
        if vtk is None:
            return

        def on_mouse_move(obj, event):
            iren = pl.iren
            if iren is None:
                return
            x, y = iren.interactor.GetEventPosition()
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.01)
            picker.Pick(x, y, 0, pl.renderer)
            actor = picker.GetActor()

            new_hover = None
            if actor is not None:
                actor_id = id(actor)
                if actor_id in self._actor_to_turtle:
                    tid = self._actor_to_turtle[actor_id]
                    for i, ts in enumerate(self._turtles):
                        if ts.turtle_id == tid:
                            new_hover = i
                            break

            if new_hover != self._hovered_idx:
                self._hovered_idx = new_hover
                self._render_hud()
                self._render_hover_tooltip()
                for ts in self._turtles:
                    if ts.visible:
                        self._render_turtle_at_step(
                            ts, self._current_step,
                        )
                pl.render()

        iren = pl.iren
        if iren is not None:
            iren.interactor.AddObserver("MouseMoveEvent", on_mouse_move)

    # ------------------------------------------------------------------
    # show()
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Build the plotter and start the visualization."""
        pl = pv.Plotter(
            title="ProspectRL Multi-Turtle Visualizer",
        )
        pl.set_background("#1a1a2e")
        self._pl = pl

        self._render_world(pl)

        for ts in self._turtles:
            self._render_turtle_at_step(ts, -1)
            self._render_trails(ts, -1)

        self._render_side_panel()
        self._render_hud()

        if self._world_size:
            size = self._world_size
        else:
            size = (40, 40, 40)
        center = np.array([size[0] / 2, size[1] / 2, size[2] / 2])
        max_dim = max(size)
        pl.camera.focal_point = center
        pl.camera.position = center + np.array([
            max_dim * 1.5, max_dim * 1.2, max_dim * 1.5,
        ])

        pl.add_legend(bcolor=(0.1, 0.1, 0.2, 0.8), face=None)
        pl.add_axes(xlabel="X", ylabel="Y (height)", zlabel="Z")

        # Key bindings
        pl.add_key_event("space", self._on_space)
        pl.add_key_event("b", self._on_back)
        pl.add_key_event("p", self._on_toggle_play)
        pl.add_key_event("plus", self._on_speed_up)
        pl.add_key_event("minus", self._on_speed_down)
        pl.add_key_event("r", self._on_reset)
        pl.add_key_event("w", self._on_world_reset)
        pl.add_key_event("d", self._on_delete)
        pl.add_key_event("h", self._on_toggle_trail)

        for i in range(9):
            pl.add_key_event(str(i + 1), self._make_select_handler(i))

        controls = (
            "Controls:\n"
            "  Space  step forward     W  reset world\n"
            "  B      step back        D  delete selected\n"
            "  P      play/pause       H  toggle trail\n"
            "  +/-    speed            1-9  select turtle\n"
            "  R      reset playback"
        )
        pl.add_text(
            controls,
            position="lower_left",
            font_size=8,
            color="#888888",
            name="controls",
        )

        if self._playing:
            self._start_timer(pl)

        self._setup_hover(pl)

        print("Opening multi-turtle viewer... (close window to exit)")
        print(f"  Turtles: {len(self._turtles)}, "
              f"Seed: {self._seed}, "
              f"Stage: {self._stage_index}")
        pl.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    n = len(CURRICULUM_STAGES)
    stage_help = [
        f"  {i}: {s.name} "
        f"({s.world_size[0]}x{s.world_size[1]}x{s.world_size[2]})"
        for i, s in enumerate(CURRICULUM_STAGES)
    ]

    parser = argparse.ArgumentParser(
        description="Multi-turtle episode visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Curriculum stages:\n" + "\n".join(stage_help),
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Path to model.zip (MaskablePPO)",
    )
    parser.add_argument(
        "--checkpoint-dir", "-d", type=str, default=None,
        help="Checkpoint directory (reads latest.json or model.zip)",
    )
    parser.add_argument(
        "--drive", type=str, default=None, nargs="?",
        const=DRIVE_CHECKPOINT_URL,
        help="Google Drive shareable folder URL",
    )
    parser.add_argument(
        "--vecnormalize", type=str, default=None,
        help="Path to vecnormalize.pkl (optional)",
    )
    parser.add_argument(
        "--stage", type=int, default=None, choices=range(n),
        help=f"Curriculum stage 0-{n - 1}",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Environment seed (default: 42)",
    )
    parser.add_argument(
        "--turtles", type=str, default=None,
        help=(
            "Comma-separated ore specs "
            "(e.g. 'diamond,iron' or 'diamond:3,iron:2'). "
            "Default: one of each ore type."
        ),
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Turtles per ore type when using defaults (default: 1)",
    )
    parser.add_argument(
        "--speed", type=int, default=200,
        help="Auto-play speed in ms/step (default: 200)",
    )
    parser.add_argument(
        "--autoplay", action="store_true",
        help="Start in auto-play mode",
    )

    args = parser.parse_args()

    # ---- Resolve model source ----
    sources = [args.drive, args.checkpoint_dir, args.model]
    n_sources = sum(s is not None for s in sources)
    if n_sources > 1:
        parser.error(
            "Provide at most one of --drive, --checkpoint-dir, or --model"
        )
    if n_sources == 0:
        args.drive = DRIVE_CHECKPOINT_URL

    model_path = args.model
    vn_path = args.vecnormalize
    stage_index = args.stage

    if args.drive:
        local_dir = _download_from_drive(args.drive)
        model_path, resolved_vn, detected_stage = _resolve_checkpoint(
            local_dir, stage_index,
        )
        if vn_path is None:
            vn_path = resolved_vn
        if stage_index is None and detected_stage is not None:
            stage_index = detected_stage
    elif args.checkpoint_dir:
        model_path, resolved_vn, detected_stage = _resolve_checkpoint(
            args.checkpoint_dir, stage_index,
        )
        if vn_path is None:
            vn_path = resolved_vn
        if stage_index is None and detected_stage is not None:
            stage_index = detected_stage
    else:
        if vn_path is None and args.model:
            sibling = Path(args.model).parent / "vecnormalize.pkl"
            if sibling.exists():
                vn_path = str(sibling)

    if stage_index is None:
        stage_index = 0
        print("No stage detected, defaulting to stage 0")

    # ---- Parse turtle specs ----
    specs = _parse_turtle_specs(args.turtles, args.count)
    print(f"Loading model from {model_path}...")
    if vn_path:
        print(f"Using vecnormalize from {vn_path}")
    print(f"Recording {len(specs)} turtle(s) on seed={args.seed}, "
          f"stage={stage_index}...")

    # ---- Build visualizer and record ----
    viz = MultiTurtleVisualizer(
        model_path=model_path,
        stage_index=stage_index,
        seed=args.seed,
        speed_ms=args.speed,
        vecnormalize_path=vn_path,
    )
    viz._playing = args.autoplay
    viz.record_all(specs)

    viz.show()


if __name__ == "__main__":
    main()
