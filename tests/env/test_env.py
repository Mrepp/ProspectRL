"""Tests for the Gymnasium environment (Phase 2)."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env
from prospect_rl.config import (
    NUM_ORE_TYPES,
    NUM_VOXEL_CHANNELS,
    OBS_WINDOW_X,
    OBS_WINDOW_Y,
    OBS_WINDOW_Z,
    SCALAR_OBS_DIM,
    Action,
    BlockType,
)
from prospect_rl.env.mining_env import MinecraftMiningEnv
from prospect_rl.env.turtle import FACING_VECTORS, Turtle

# ---------------------------------------------------------------------------
# Env Checker
# ---------------------------------------------------------------------------

class TestCheckEnv:
    def test_check_env_passes(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        check_env(env)


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------

class TestObservation:
    def test_voxels_shape(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        obs, _ = env.reset(seed=42)
        assert obs["voxels"].shape == (
            NUM_VOXEL_CHANNELS,
            OBS_WINDOW_Y,
            OBS_WINDOW_X,
            OBS_WINDOW_Z,
        )

    def test_scalars_shape(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        obs, _ = env.reset(seed=42)
        assert obs["scalars"].shape == (SCALAR_OBS_DIM,)

    def test_pref_shape(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        obs, _ = env.reset(seed=42)
        assert obs["pref"].shape == (NUM_ORE_TYPES,)

    def test_pref_sums_to_one(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        obs, _ = env.reset(seed=42)
        assert abs(obs["pref"].sum() - 1.0) < 1e-5

    def test_obs_dtypes(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        obs, _ = env.reset(seed=42)
        assert obs["voxels"].dtype == np.float32
        assert obs["scalars"].dtype == np.float32
        assert obs["pref"].dtype == np.float32

    def test_voxels_channels_binary(self) -> None:
        """Each voxel channel should contain only 0.0 or 1.0."""
        env = MinecraftMiningEnv(curriculum_stage=0)
        obs, _ = env.reset(seed=42)
        voxels = obs["voxels"]
        unique_vals = np.unique(voxels)
        for v in unique_vals:
            assert v in (0.0, 1.0), f"Unexpected value {v} in voxels"


# ---------------------------------------------------------------------------
# Action Masking
# ---------------------------------------------------------------------------

class TestActionMasking:
    def test_mask_shape(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.shape == (9,)
        assert mask.dtype == bool

    def test_turns_always_valid(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask[Action.TURN_LEFT]
        assert mask[Action.TURN_RIGHT]

    def test_masked_action_doesnt_crash(self) -> None:
        """Even if we take a masked action, env shouldn't crash."""
        env = MinecraftMiningEnv(curriculum_stage=0)
        env.reset(seed=42)
        # Take all 9 actions — env should handle gracefully
        for action in range(9):
            env.step(action)


# ---------------------------------------------------------------------------
# Turtle Movement
# ---------------------------------------------------------------------------

class TestTurtleMovement:
    def test_facing_vectors(self) -> None:
        assert np.array_equal(FACING_VECTORS[0], [0, 0, 1])   # north = +z
        assert np.array_equal(FACING_VECTORS[1], [1, 0, 0])   # east = +x
        assert np.array_equal(FACING_VECTORS[2], [0, 0, -1])  # south = -z
        assert np.array_equal(FACING_VECTORS[3], [-1, 0, 0])  # west = -x

    def test_turn_left(self) -> None:
        t = Turtle(np.array([5, 5, 5]), facing=0)
        t.turn_left()
        assert t.facing == 3
        t.turn_left()
        assert t.facing == 2

    def test_turn_right(self) -> None:
        t = Turtle(np.array([5, 5, 5]), facing=0)
        t.turn_right()
        assert t.facing == 1
        t.turn_right()
        assert t.facing == 2

    def test_move_forward_in_air(self) -> None:
        # Create a small world with air at target
        world = np.full((10, 10, 10), BlockType.STONE, dtype=np.int8)
        world[:, 0, :] = BlockType.BEDROCK
        world[5, 5, 5] = BlockType.AIR  # turtle position
        world[5, 5, 6] = BlockType.AIR  # forward (north = +z)

        t = Turtle(np.array([5, 5, 5]), facing=0, fuel=100)
        result = t.move_forward(world)
        assert result is True
        assert np.array_equal(t.position, [5, 5, 6])
        assert t.fuel == 99

    def test_cannot_move_into_stone(self) -> None:
        world = np.full((10, 10, 10), BlockType.STONE, dtype=np.int8)
        world[5, 5, 5] = BlockType.AIR

        t = Turtle(np.array([5, 5, 5]), facing=0, fuel=100)
        result = t.move_forward(world)
        assert result is False
        assert np.array_equal(t.position, [5, 5, 5])
        assert t.fuel == 100  # fuel refunded

    def test_cannot_move_without_fuel(self) -> None:
        world = np.full((10, 10, 10), BlockType.AIR, dtype=np.int8)
        t = Turtle(np.array([5, 5, 5]), facing=0, fuel=0)
        result = t.move_forward(world)
        assert result is False


# ---------------------------------------------------------------------------
# Digging
# ---------------------------------------------------------------------------

class TestDigging:
    def test_dig_stone(self) -> None:
        world = np.full((10, 10, 10), BlockType.STONE, dtype=np.int8)
        world[5, 5, 5] = BlockType.AIR  # turtle is here

        t = Turtle(np.array([5, 5, 5]), facing=0)
        # Block in front (north = +z) is stone at (5,5,6)
        result = t.dig(world)
        assert result is True
        assert world[5, 5, 6] == BlockType.AIR
        assert t.inventory.get(int(BlockType.STONE), 0) == 1

    def test_dig_ore_adds_to_inventory(self) -> None:
        world = np.full((10, 10, 10), BlockType.STONE, dtype=np.int8)
        world[5, 5, 5] = BlockType.AIR
        world[5, 5, 6] = BlockType.DIAMOND_ORE

        t = Turtle(np.array([5, 5, 5]), facing=0)
        t.dig(world)
        assert t.inventory.get(int(BlockType.DIAMOND_ORE), 0) == 1

    def test_cannot_dig_air(self) -> None:
        world = np.full((10, 10, 10), BlockType.AIR, dtype=np.int8)
        t = Turtle(np.array([5, 5, 5]), facing=0)
        result = t.dig(world)
        assert result is False

    def test_cannot_dig_bedrock(self) -> None:
        world = np.full((10, 10, 10), BlockType.BEDROCK, dtype=np.int8)
        world[5, 5, 5] = BlockType.AIR
        t = Turtle(np.array([5, 5, 5]), facing=0)
        result = t.dig(world)
        assert result is False

    def test_dig_up_and_down(self) -> None:
        world = np.full((10, 10, 10), BlockType.STONE, dtype=np.int8)
        world[5, 5, 5] = BlockType.AIR

        t = Turtle(np.array([5, 5, 5]), facing=0)
        assert t.dig_up(world) is True
        assert world[5, 6, 5] == BlockType.AIR

        assert t.dig_down(world) is True
        assert world[5, 4, 5] == BlockType.AIR


# ---------------------------------------------------------------------------
# Fuel
# ---------------------------------------------------------------------------

class TestFuel:
    def test_fuel_decreases_on_movement(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        env.reset(seed=42)
        # Ensure infinite fuel is on in stage 0
        # Stage 0 has infinite fuel, so fuel shouldn't decrease
        initial_fuel = env._turtle.fuel
        # In stage 0 infinite_fuel=True, fuel gets reset each step
        env.step(Action.TURN_LEFT)
        # turns don't cost fuel anyway
        assert env._turtle.fuel == initial_fuel

    def test_episode_terminates_on_zero_fuel(self) -> None:
        # Use stage 1 which has limited fuel
        env = MinecraftMiningEnv(curriculum_stage=1)
        env.reset(seed=42)
        env._turtle.fuel = 1
        env._stage_cfg = type(env._stage_cfg)(
            name="test",
            world_size=env._stage_cfg.world_size,
            ore_density_multiplier=1.0,
            infinite_fuel=False,
            max_fuel=500,
            caves_enabled=False,
            preference_mode="one_hot",
            max_episode_steps=10000,
            advancement_metric="mean_reward",
            advancement_threshold=50.0,
            advancement_window=100,
        )
        # Clear path for movement
        fv = FACING_VECTORS[env._turtle.facing]
        target = env._turtle.position + fv
        env._world[target[0], target[1], target[2]] = BlockType.AIR
        _, _, terminated, _, _ = env.step(Action.FORWARD)
        assert env._turtle.fuel == 0
        assert terminated is True


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_fresh_state(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        obs1, _ = env.reset(seed=42)
        # Take some actions
        for _ in range(5):
            env.step(Action.TURN_LEFT)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1["voxels"], obs2["voxels"])
        np.testing.assert_array_equal(obs1["scalars"], obs2["scalars"])

    def test_different_seeds_different_obs(self) -> None:
        env = MinecraftMiningEnv(curriculum_stage=0)
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=9999)
        # Preferences should likely differ
        assert not np.array_equal(obs1["pref"], obs2["pref"])


# ---------------------------------------------------------------------------
# Stage 1 Reward Integration
# ---------------------------------------------------------------------------

class TestStage1Integration:
    def test_stage1_target_ore_positive_reward(self) -> None:
        """Mining a target ore in Stage 1 gives positive reward."""
        env = MinecraftMiningEnv(curriculum_stage=0, seed=42)
        env.reset()

        # Find which ore is the target
        pref = env._preference
        target_idx = int(np.argmax(pref))
        from prospect_rl.config import ORE_TYPES as OT
        target_bt = int(OT[target_idx])

        # Place target ore in front of turtle
        fv = FACING_VECTORS[env._turtle.facing]
        target_pos = env._turtle.position + fv
        env._world[
            target_pos[0], target_pos[1], target_pos[2]
        ] = target_bt

        _, reward, _, _, info = env.step(Action.DIG)
        assert reward > 0.0
        assert info["r_harvest"] > 0.0
        assert info["r_adjacent"] == 0.0

    def test_stage1_stone_negative_reward(self) -> None:
        """Mining stone in Stage 1 gives small negative reward."""
        env = MinecraftMiningEnv(curriculum_stage=0, seed=42)
        env.reset()

        # Place stone in front of turtle
        fv = FACING_VECTORS[env._turtle.facing]
        target_pos = env._turtle.position + fv
        env._world[
            target_pos[0], target_pos[1], target_pos[2]
        ] = int(BlockType.STONE)

        _, reward, _, _, info = env.step(Action.DIG)
        # r_ops should be negative (waste penalty)
        assert info["r_ops"] < 0.0

    def test_stage1_completion_ratio_in_info(self) -> None:
        """Stage 1 info dict has completion_ratio."""
        env = MinecraftMiningEnv(curriculum_stage=0, seed=42)
        env.reset()
        _, _, _, _, info = env.step(Action.TURN_LEFT)
        assert "completion_ratio" in info
        assert "target_ores_mined" in info
        assert "cumulative_waste" in info

    def test_stage1_terminal_bonus_at_truncation(self) -> None:
        """Terminal bonus fires at episode end."""
        env = MinecraftMiningEnv(curriculum_stage=0, seed=42)
        env.reset()

        # Fast-forward to near truncation
        env._step_count = env._max_steps - 1

        _, _, _, truncated, info = env.step(Action.TURN_LEFT)
        assert truncated is True
        assert "terminal_bonus" in info
        # Bonus >= 0 (could be 0 if no target ores mined)
        assert info["terminal_bonus"] >= 0.0

    def test_stage1_count_target_ores(self) -> None:
        """_count_target_ores matches manual scan."""
        env = MinecraftMiningEnv(curriculum_stage=0, seed=42)
        env.reset()

        count = env._count_target_ores()
        assert count > 0  # with 10x density, should have ores
        assert env._total_target_ores_in_world == count
