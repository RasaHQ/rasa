import types
import uuid
from datetime import datetime
from typing import Dict
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from rasa.builder.config import OPENAI_MODEL
from rasa.builder.copilot.models import EventContent, ResponseCategory
from rasa.builder.telemetry.copilot_segment_telemetry import (
    COPILOT_BOT_MESSAGE_EVENT,
    COPILOT_USER_MESSAGE_EVENT,
    CopilotSegmentTelemetry,
)


class TrackedEvent(BaseModel):
    event: str
    user_id: str
    properties: Dict


@pytest.fixture
def telemetry_events(monkeypatch) -> tuple[CopilotSegmentTelemetry, list[TrackedEvent]]:
    """Telemetry instance with _track patched to capture payloads."""
    events: list[TrackedEvent] = []

    def fake_track(event: str, user_id: str, properties: dict) -> None:
        events.append(TrackedEvent(event=event, user_id=user_id, properties=properties))

    monkeypatch.setattr(
        "rasa.builder.telemetry.copilot_segment_telemetry._track", fake_track
    )
    telemetry = CopilotSegmentTelemetry(project_id="proj-123", user_id="user-xyz")
    return telemetry, events


def test_log_user_turn_sends_minimal_fields(
    telemetry_events: tuple[CopilotSegmentTelemetry, list[TrackedEvent]],
):
    telemetry, events = telemetry_events
    telemetry.log_user_turn("hi there")

    event = events[0]
    assert event.event == COPILOT_USER_MESSAGE_EVENT
    assert event.user_id == "user-xyz"
    assert event.properties["project_id"] == "proj-123"
    assert event.properties["text"] == "hi there"
    uuid.UUID(event.properties["message_id"])
    datetime.fromisoformat(event.properties["timestamp"])


def test_extract_flags():
    response_categories = {
        ResponseCategory.ROLEPLAY_DETECTION,
        ResponseCategory.COPILOT,
        ResponseCategory.OUT_OF_SCOPE_DETECTION,
        ResponseCategory.ROLEPLAY_DETECTION,
    }
    handler = types.SimpleNamespace(
        generated_responses=[
            MagicMock(response_category=category) for category in response_categories
        ]
    )

    flags = CopilotSegmentTelemetry._extract_flags(handler)
    assert set(flags) == {category.value for category in response_categories}


def test_full_text_concatenates_only_non_empty_content():
    """_full_text should concatenate every non-None `content` in order."""
    handler = types.SimpleNamespace(
        generated_responses=[
            MagicMock(response_category=ResponseCategory.COPILOT, content="Hello"),
            MagicMock(response_category=ResponseCategory.COPILOT, content=None),
            MagicMock(response_category=ResponseCategory.COPILOT, content=" World"),
            MagicMock(response_category=ResponseCategory.REFERENCE, content=""),
        ]
    )

    assert CopilotSegmentTelemetry._full_text(handler) == "Hello World"


def test_log_copilot_turn_emits_complete_payload(
    telemetry_events: tuple[CopilotSegmentTelemetry, list[TrackedEvent]],
):
    telemetry, events = telemetry_events
    telemetry.log_copilot_turn(
        text="answer",
        source_urls=["https://rasa.com"],
        flags=["roleplay_detection"],
        model=OPENAI_MODEL,
        latency_ms=123,
        input_tokens=10,
        cached_prompt_tokens=5,
        output_tokens=20,
        total_tokens=30,
    )

    event = events[0]
    assert event.event == COPILOT_BOT_MESSAGE_EVENT
    assert event.properties["model"] == OPENAI_MODEL
    assert event.properties["latency_ms"] == 123
    assert event.properties["source_urls"] == ["https://rasa.com"]
    assert event.properties["flags"] == ["roleplay_detection"]
    assert event.properties["input_tokens"] == 10
    assert event.properties["cached_prompt_tokens"] == 5
    assert event.properties["output_tokens"] == 20
    assert event.properties["total_tokens"] == 30


def test_log_copilot_from_handler_combines_everything(
    telemetry_events: tuple[CopilotSegmentTelemetry, list[TrackedEvent]],
):
    # Given
    system_message = {"role": "system", "content": "system message"}
    chat_history = [{"role": "user", "content": "chat history"}]
    last_user_message = "last user message"

    telemetry, events = telemetry_events

    # MagicMock lets us attach arbitrary attributes required by the code
    handler = types.SimpleNamespace(
        generated_responses=[
            MagicMock(
                response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                content="oops",
            ),
            MagicMock(response_category=ResponseCategory.COPILOT, content=" final"),
        ]
    )

    docs = [MagicMock(url="https://example.org/doc")]

    # Create sample tracker event attachments
    tracker_event_attachments = [
        EventContent(
            type="event",
            event="user",
            text="Hello, I need help with my bot",
            timestamp="2024-01-01T10:00:00Z",
        ),
        EventContent(
            type="event",
            event="action",
            action_name="utter_greet",
            timestamp="2024-01-01T10:00:01Z",
        ),
    ]

    telemetry.log_copilot_from_handler(
        handler=handler,
        used_documents=docs,
        latency_ms=321,
        model=OPENAI_MODEL,
        prompt_tokens=10,
        cached_prompt_tokens=5,
        completion_tokens=20,
        total_tokens=30,
        system_message=system_message,
        chat_history=chat_history,
        last_user_message=last_user_message,
        tracker_event_attachments=tracker_event_attachments,
    )

    # Third recorded call: index 2
    event = events[0]
    assert event.event == COPILOT_BOT_MESSAGE_EVENT
    assert event.properties["text"] == "oops final"
    assert event.properties["source_urls"] == ["https://example.org/doc"]
    assert sorted(event.properties["flags"]) == ["copilot", "out_of_scope_detection"]
    assert event.properties["latency_ms"] == 321
    assert event.properties["input_tokens"] == 10
    assert event.properties["cached_prompt_tokens"] == 5
    assert event.properties["output_tokens"] == 20
    assert event.properties["total_tokens"] == 30
    # assert event.properties["system_message"] == system_message
    assert event.properties["chat_history"] == chat_history
    assert event.properties["last_user_message"] == last_user_message
    # Verify tracker event attachments are included
    assert "tracker_event_attachments" in event.properties
    assert len(event.properties["tracker_event_attachments"]) == 2
    # Check that the attachments are properly serialized
    attachments = event.properties["tracker_event_attachments"]
    assert attachments[0]["event"] == "user"
    assert attachments[0]["event_data"]["text"] == "Hello, I need help with my bot"
    assert attachments[0]["event_data"]["timestamp"] == "2024-01-01T10:00:00Z"
    assert attachments[1]["event"] == "action"
    assert attachments[1]["event_data"]["action_name"] == "utter_greet"
    assert attachments[1]["event_data"]["timestamp"] == "2024-01-01T10:00:01Z"
