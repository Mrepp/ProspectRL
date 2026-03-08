"""Inter-agent communication system.

Simple message passing for coordination. Agents can broadcast
discoveries and status to neighbors within communication range.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum


class MessageType(IntEnum):
    """Types of inter-agent messages."""

    ORE_FOUND = 0       # discovered target ore at position
    REGION_EMPTY = 1    # region is depleted
    HAZARD = 2          # dangerous area (lava, deep drop)
    FUEL_LOW = 3        # agent running low on fuel


@dataclass(slots=True)
class AgentMessage:
    """A message from one agent to others.

    Parameters
    ----------
    sender_id:
        Agent that sent the message.
    message_type:
        What kind of information.
    position:
        World coordinates relevant to the message.
    ore_type:
        Ore type index if relevant (ORE_FOUND), else None.
    timestamp:
        Step number when the message was created.
    """

    sender_id: int
    message_type: MessageType
    position: tuple[int, int, int]
    ore_type: int | None = None
    timestamp: int = 0


class MessageBuffer:
    """Per-agent message inbox with bounded capacity.

    Parameters
    ----------
    max_messages:
        Maximum messages to retain. Oldest are dropped when full.
    """

    def __init__(self, max_messages: int = 50) -> None:
        self._inbox: deque[AgentMessage] = deque(maxlen=max_messages)
        self._outbox: list[AgentMessage] = []

    def receive(self, message: AgentMessage) -> None:
        """Add a message to the inbox."""
        self._inbox.append(message)

    def send(self, message: AgentMessage) -> None:
        """Queue a message for broadcast."""
        self._outbox.append(message)

    def get_inbox(self) -> list[AgentMessage]:
        """Return all messages in inbox (does not clear)."""
        return list(self._inbox)

    def flush_outbox(self) -> list[AgentMessage]:
        """Return and clear outbox messages."""
        msgs = self._outbox
        self._outbox = []
        return msgs

    def clear(self) -> None:
        """Clear both inbox and outbox."""
        self._inbox.clear()
        self._outbox.clear()

    @property
    def inbox_count(self) -> int:
        return len(self._inbox)
