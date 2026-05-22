"""Unit tests for `rasa.core.channels.conversation_queue.utils`."""

from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from rasa.core.channels.channel import OutputChannel
from rasa.core.channels.constants import (
    USER_CONVERSATION_SESSION_END,
    USER_CONVERSATION_SESSION_START,
    USER_CONVERSATION_SILENCE_TIMEOUT,
)
from rasa.core.channels.conversation_queue.events import (
    BargeInInputEvent,
    DTMFInputEvent,
    FinalTranscriptInputEvent,
    InputEvent,
    SessionEndedInputEvent,
    SessionStartedInputEvent,
    SilenceDetectedInputEvent,
)
from rasa.core.channels.conversation_queue.utils import (
    coalesce_final_transcripts,
)
from rasa.core.channels.voice_ready.utils import CallParameters

_CALL_PARAMETERS = CallParameters(
    call_id="call-1",
    user_phone="+15551000",
    bot_phone="+15552000",
    direction="inbound",
    stream_id="stream-1",
    language="en",
)
_INPUT_CHANNEL = "twilio_media_streams"


class _UnknownInputEvent(InputEvent):
    """A non-mapped InputEvent used to verify `None` return paths."""


@pytest.mark.parametrize(
    "given_events,expected_events",
    [
        pytest.param(
            [],
            [],
            id="empty",
        ),
        pytest.param(
            [BargeInInputEvent(), SessionEndedInputEvent()],
            [BargeInInputEvent(), SessionEndedInputEvent()],
            id="no_transcripts_preserves_events",
        ),
        pytest.param(
            [
                FinalTranscriptInputEvent(text="hello", metadata={"asr": 0.9}),
                FinalTranscriptInputEvent(text="there", metadata={"asr": 0.95}),
                FinalTranscriptInputEvent(text="world", metadata={"asr": 0.99}),
            ],
            [
                FinalTranscriptInputEvent(
                    text="hello there world", metadata={"asr": 0.99}
                )
            ],
            id="merges_consecutive_transcripts",
        ),
        pytest.param(
            [
                BargeInInputEvent(),
                FinalTranscriptInputEvent(text="hello"),
                FinalTranscriptInputEvent(text="there"),
                SessionEndedInputEvent(),
            ],
            [
                BargeInInputEvent(),
                FinalTranscriptInputEvent(text="hello there"),
                SessionEndedInputEvent(),
            ],
            id="preserves_position_of_other_events",
        ),
        pytest.param(
            [
                SessionStartedInputEvent(metadata=asdict(_CALL_PARAMETERS)),
                FinalTranscriptInputEvent(text="a"),
                FinalTranscriptInputEvent(text="b"),
                FinalTranscriptInputEvent(text="c"),
                BargeInInputEvent(),
                FinalTranscriptInputEvent(text="d"),
                FinalTranscriptInputEvent(text="e"),
                DTMFInputEvent(text="123"),
                FinalTranscriptInputEvent(text="f"),
                FinalTranscriptInputEvent(text="g"),
                SessionEndedInputEvent(),
            ],
            [
                SessionStartedInputEvent(metadata=asdict(_CALL_PARAMETERS)),
                FinalTranscriptInputEvent(text="a b c"),
                BargeInInputEvent(),
                FinalTranscriptInputEvent(text="d e"),
                DTMFInputEvent(text="123"),
                FinalTranscriptInputEvent(text="f g"),
                SessionEndedInputEvent(),
            ],
            id="handles_multiple_runs",
        ),
    ],
)
def test_coalesce_final_transcripts(
    given_events: list,
    expected_events: list,
) -> None:
    assert coalesce_final_transcripts(given_events) == expected_events


@pytest.mark.parametrize(
    "event,expected_text,expected_metadata",
    [
        (
            SessionStartedInputEvent(metadata=asdict(_CALL_PARAMETERS)),
            USER_CONVERSATION_SESSION_START,
            asdict(_CALL_PARAMETERS),
        ),
        (SessionEndedInputEvent(), USER_CONVERSATION_SESSION_END, None),
        (SilenceDetectedInputEvent(), USER_CONVERSATION_SILENCE_TIMEOUT, None),
        (DTMFInputEvent(text="42"), "42", {"source": "dtmf"}),
    ],
)
def test_event_to_user_message_builds_expected_payload(
    event: InputEvent,
    expected_text: str,
    expected_metadata: dict,
) -> None:
    output_channel = MagicMock(spec=OutputChannel)

    message = event.to_user_message(
        output_channel,
        sender_id="sender-1",
        input_channel=_INPUT_CHANNEL,
    )

    assert message is not None
    assert message.text == expected_text
    assert message.metadata == expected_metadata
    assert message.sender_id == "sender-1"
    assert message.input_channel == _INPUT_CHANNEL
    assert message.output_channel is output_channel


@pytest.mark.parametrize(
    "event",
    [
        BargeInInputEvent(),
        _UnknownInputEvent(),
    ],
)
def test_event_to_user_message_returns_none_for_non_user_message_events(
    event: InputEvent,
) -> None:
    output_channel = MagicMock(spec=OutputChannel)
    assert (
        event.to_user_message(
            output_channel,
            sender_id="sender-1",
            input_channel=_INPUT_CHANNEL,
        )
        is None
    )
