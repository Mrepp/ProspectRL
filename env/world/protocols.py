"""Protocol interfaces for world and turtle backends.

These protocols define the minimal API that any world or turtle
implementation must satisfy to work with ``MinecraftMiningEnv``.

Existing classes already satisfy these protocols via duck typing:
- ``World``, ``RealChunkWorld``, ``_StubWorld`` → ``WorldBackend``
- ``Turtle`` → ``TurtleBackend``

Future implementations (e.g. ``MinecraftWorldBackend`` for live
MC servers) should implement these protocols explicitly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class WorldBackend(Protocol):
    """Protocol for 3D block world backends.

    Any class implementing this protocol can be used by
    ``MinecraftMiningEnv`` as its world backend.
    """

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(sx, sy, sz)`` world dimensions."""
        ...

    def get_block(self, x: int, y: int, z: int) -> int:
        """Return the block type at ``(x, y, z)``."""
        ...

    def set_block(self, x: int, y: int, z: int, block_id: int) -> None:
        """Set the block at ``(x, y, z)``."""
        ...

    def count_blocks(self, block_ids: list[int]) -> int:
        """Count blocks matching any of the given IDs."""
        ...

    def get_sliding_window(
        self,
        pos: np.ndarray,
        radius_xz: int = 4,
        y_above: int = 8,
        y_below: int = 23,
        fill_value: int = 3,  # BlockType.BEDROCK
    ) -> np.ndarray:
        """Extract a 3D observation window centred on *pos*.

        Returns shape ``(window_x, window_y, window_z)`` as int8.
        """
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...


@runtime_checkable
class TurtleBackend(Protocol):
    """Protocol for turtle action execution backends.

    The default ``Turtle`` class executes actions against a local
    ``WorldBackend`` (modifying the numpy array directly).  A future
    ``MinecraftTurtleBackend`` would send commands via WebSocket.
    """

    position: np.ndarray
    facing: int
    fuel: int
    max_fuel: int
    inventory: dict[int, int]

    def move_forward(self, world: Any) -> bool:
        ...

    def move_up(self, world: Any) -> bool:
        ...

    def move_down(self, world: Any) -> bool:
        ...

    def turn_left(self) -> bool:
        ...

    def turn_right(self) -> bool:
        ...

    def dig(self, world: Any) -> bool:
        ...

    def dig_up(self, world: Any) -> bool:
        ...

    def dig_down(self, world: Any) -> bool:
        ...
