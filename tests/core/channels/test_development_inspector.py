import pathlib

import pytest

import rasa.core.run
from rasa.core.agent import Agent
from rasa.core.channels.development_inspector import (
    INSPECT_LEGACY_TEMPLATE_PATH,
    DevelopmentInspectProxy,
)
from rasa.core.channels.rest import RestInput
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    DialogueStackUpdated,
    FlowCompleted,
    FlowStarted,
    SessionStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker

ABSOLUTE_INSPECT_FOLDER_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "rasa"
    / "core"
    / "channels"
    / INSPECT_LEGACY_TEMPLATE_PATH
)

ABSOLUTE_INSPECT_TEMPLATE_PATH = ABSOLUTE_INSPECT_FOLDER_PATH / "index.html"


@pytest.fixture
def mock_tracker_stream():
    """Fixture that provides a mock tracker stream for testing."""

    class MockTrackerStream:
        def __init__(self):
            self.called = False

        async def broadcast(self, message: str):
            self.called = True

    return MockTrackerStream()


def test_inspect_html_path() -> None:
    channel = DevelopmentInspectProxy(RestInput.from_credentials({}))
    assert channel.inspect_html_path() == str(ABSOLUTE_INSPECT_FOLDER_PATH)


def test_blueprint_inspect() -> None:
    input_channel = DevelopmentInspectProxy(RestInput.from_credentials({}))

    app = rasa.core.run.configure_app([input_channel], port=5004)
    app.ctx.agent = Agent()
    _, res = app.test_client.get("/webhooks/rest/inspect.html")

    assert res.status_code == 200
    # binary comparison to be platform-agnostic
    with open(ABSOLUTE_INSPECT_TEMPLATE_PATH, mode="rb") as handle:
        assert res.body == handle.read()


async def test_on_tracker_updated(
    tracker_with_restarted_event: DialogueStateTracker, mock_tracker_stream
):
    inspector = DevelopmentInspectProxy(RestInput.from_credentials({}))
    inspector.tracker_stream = mock_tracker_stream

    # Call on_tracker_updated
    await inspector.on_tracker_updated(tracker_with_restarted_event)
    assert mock_tracker_stream.called


async def test_on_tracker_updated_with_events(mock_tracker_stream):
    events = [
        ActionExecuted(action_name="action_session_start", policy=None, confidence=1.0),
        SessionStarted(),
        ActionExecuted(action_name="action_listen", policy=None, confidence=None),
        UserUttered("/session_start"),
        SlotSet(
            key="flow_hashes",
            value={
                "welcome": "638fe5c911c18eedcb5335d6b649b6ab",
                "pattern_search": "03ead4a9fe24abae4caa0e119717cf6a",
            },
        ),
        DialogueStackUpdated(
            update='[{"op": "add", "path": "/0", "value": {"frame_id": "XD8RY3UY", "flow_id": "pattern_session_start", "step_id": "START", "type": "pattern_session_start"}}]'  # noqa: E501
        ),
        FlowStarted(flow_id="pattern_session_start"),
        ActionExecuted(
            action_name="utter_greeting", policy="FlowPolicy", confidence=1.0
        ),
        BotUttered(
            "To protect your personal data, please do share sensitive member information here, like your password, or SSN. Hi, I'm Freedom Insurance's high-trust AI assistant. I can help you with claims, questions about your coverage, billing questions and more. What would you like to do today?",  # noqa: E501
            {
                "elements": None,
                "quick_replies": None,
                "buttons": None,
                "attachment": None,
                "image": None,
                "custom": None,
            },
            {
                "metadata": {"rephrase": False},
                "active_flow": "pattern_session_start",
                "step_id": "pattern_session_start_0_utter_greeting",
                "utter_action": "utter_greeting",
                "utter_source": "ContextualResponseRephraser",
                "domain_ground_truth": [
                    "To protect your personal data, please do share sensitive member information here, like your password, or SSN.\n\nHi, I'm Freedom Insurance's high-trust AI assistant. I can help you with claims, questions about your coverage, billing questions and more. What would you like to do today?"  # noqa: E501
                ],
                "model_id": "0ddb2e75dedf438585d6a78a0a003faf",
                "assistant_id": "freedom_insurance",
            },
            1742889184.3784692,
        ),
        DialogueStackUpdated(
            update='[{"op": "replace", "path": "/0/step_id", "value": "END"}]'
        ),
        DialogueStackUpdated(update='[{"op": "remove", "path": "/0"}]'),
        FlowCompleted(
            flow_id="pattern_session_start",
            step_id="pattern_session_start_0_utter_greeting",
        ),
        ActionExecuted(
            action_name="action_listen", policy="FlowPolicy", confidence=1.0
        ),
    ]
    test_tracker = DialogueStateTracker.from_events("test", events)
    inspector = DevelopmentInspectProxy(RestInput.from_credentials({}))
    inspector.tracker_stream = mock_tracker_stream

    await inspector.on_tracker_updated(test_tracker)
    assert mock_tracker_stream.called


async def test_on_streaming_response(mock_tracker_stream):
    """Test that streaming response broadcasts include accumulated text."""
    events = [
        SessionStarted(),
        UserUttered("Hello"),
    ]
    test_tracker = DialogueStateTracker.from_events("test", events)
    inspector = DevelopmentInspectProxy(RestInput.from_credentials({}))
    inspector.tracker_stream = mock_tracker_stream

    await inspector.on_streaming_response(test_tracker, "Hello, how")
    assert mock_tracker_stream.called


@pytest.fixture
def mock_tracker_stream_with_capture():
    """Fixture that provides a mock tracker stream that captures messages."""
    import json

    class MockTrackerStreamWithCapture:
        def __init__(self):
            self.messages = []

        async def broadcast(self, message: str):
            self.messages.append(json.loads(message))

    return MockTrackerStreamWithCapture()


async def test_on_streaming_response_includes_synthetic_bot_event(
    mock_tracker_stream_with_capture,
):
    """Test that streaming response adds a synthetic bot event with streaming flag."""
    events = [
        SessionStarted(),
        UserUttered("Hello"),
    ]
    test_tracker = DialogueStateTracker.from_events("test", events)
    inspector = DevelopmentInspectProxy(RestInput.from_credentials({}))
    inspector.tracker_stream = mock_tracker_stream_with_capture

    await inspector.on_streaming_response(test_tracker, "Hello, how are you")

    assert len(mock_tracker_stream_with_capture.messages) == 1
    message = mock_tracker_stream_with_capture.messages[0]

    # Check that the last event is the synthetic streaming bot event
    last_event = message["events"][-1]
    assert last_event["event"] == "bot"
    assert last_event["text"] == "Hello, how are you"
    assert last_event["metadata"]["streaming"] is True
