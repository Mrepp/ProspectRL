"""Telemetry event system for multi-agent observation sharing.

Turtles have limited sensing (3 blocks/step via CC:Tweaked inspect).
All world knowledge flows through telemetry events that update the
shared belief map.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TelemetryEventType(IntEnum):
    """Types of telemetry events emitted by agents."""

    BLOCK_OBSERVED = 0   # normal 3-block inspection
    BLOCK_REMOVED = 1    # agent mined a block (dig action)
    BLOCK_ADDED = 2      # block appeared where air was expected (gravel, player)
    BLOCK_CHANGED = 3    # block type changed unexpectedly
    PATH_BLOCKED = 4     # A* path step failed — expected air, found solid/agent


@dataclass(slots=True)
class TelemetryEvent:
    """A single telemetry event from an agent.

    Parameters
    ----------
    event_type:
        What happened.
    agent_id:
        Which agent produced this event.
    position:
        World coordinates of the affected block.
    block_type:
        ``BlockType`` value observed at position.
    previous_belief:
        What P(ore) was at this position before the event.
    timestamp:
        Step number when the event occurred.
    ore_type:
        If ore was found/mined, which type index (0-7). None otherwise.
    """

    event_type: TelemetryEventType
    agent_id: int
    position: tuple[int, int, int]
    block_type: int
    previous_belief: float
    timestamp: int
    ore_type: int | None = None
