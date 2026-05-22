"""Typed input events for queue-based channel communication."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rasa.core.channels.channel import OutputChannel, UserMessage
from rasa.core.channels.constants import (
    USER_CONVERSATION_SESSION_END,
    USER_CONVERSATION_SESSION_START,
    USER_CONVERSATION_SILENCE_TIMEOUT,
)


class InputEvent(ABC):
    """Base class for all input events sent from channels to Rasa via queue."""

    def to_user_message(
        self,
        output_channel: OutputChannel,
        sender_id: str,
        input_channel: str,
    ) -> Optional[UserMessage]:
        """Build a `UserMessage` counterpart for this event, if there is one."""
        return None


class VoiceInputEvent(InputEvent):
    """Base class for all voice-specific input events."""

    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_user_message(
        self,
        output_channel: OutputChannel,
        sender_id: str,
        input_channel: str,
    ) -> Optional[UserMessage]:
        """Build the voice `UserMessage` counterpart for this event, if any."""
        if self.text is None:
            return None

        return UserMessage(
            text=self.text,
            output_channel=output_channel,
            sender_id=sender_id,
            input_channel=input_channel,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class SessionStartedInputEvent(VoiceInputEvent):
    """Marks the beginning of a voice session.

    Sent when a WebSocket connects and call parameters are established.

    Attributes:
        text: constant string
        metadata: The call parameters for this session.
    """

    metadata: Dict[str, Any]
    text: str = field(default=USER_CONVERSATION_SESSION_START, init=False)


@dataclass(frozen=True)
class FinalTranscriptInputEvent(VoiceInputEvent):
    """A finalised ASR transcript for one voice turn.

    Attributes:
        text: The final transcript text from ASR.
        metadata: Optional metadata from the channel (e.g. ASR confidence).
    """

    text: str
    metadata: Optional[Dict[str, Any]] = field(default=None, compare=False)


@dataclass(frozen=True)
class DTMFInputEvent(VoiceInputEvent):
    """Collection of one or more DTMF digits.

    The collection still happens in the input channel before creating this event.

    Attributes:
        text: The DTMF digit(s) pressed by the user.
    """

    text: str
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {"source": "dtmf"},
        compare=False,
    )


@dataclass(frozen=True)
class SilenceDetectedInputEvent(VoiceInputEvent):
    """Channel signals a period of silence from the user."""

    text: str = field(default=USER_CONVERSATION_SILENCE_TIMEOUT, init=False)
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class BargeInInputEvent(VoiceInputEvent):
    """Channel signals that the user started speaking while the bot was speaking.

    Does not record at what point in the bot message the barge-in occurred.
    """

    pass


@dataclass(frozen=True)
class SessionEndedInputEvent(VoiceInputEvent):
    """Marks the end of a voice session.

    Sent when the call terminates or the WebSocket disconnects.
    """

    text: str = field(default=USER_CONVERSATION_SESSION_END, init=False)
    metadata: Optional[Dict[str, Any]] = None
