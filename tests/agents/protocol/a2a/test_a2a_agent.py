import asyncio
from typing import Any, AsyncGenerator, Callable, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from a2a.client import A2AClientError, A2AClientHTTPError
from a2a.client.errors import A2AClientJSONRPCError
from a2a.types import (
    Artifact,
    DataPart,
    FilePart,
    FileWithUri,
    InternalError,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from structlog.testing import capture_logs

from rasa.agents.constants import (
    A2A_AGENT_CONTEXT_ID_KEY,
    A2A_AGENT_TASK_ID_KEY,
    AGENT_DEFAULT_MAX_RETRIES,
    AGENT_DEFAULT_TIMEOUT_SECONDS,
    AGENT_METADATA_STRUCTURED_RESULTS_KEY,
)
from rasa.agents.core.types import AgentStatus
from rasa.agents.protocol.a2a.a2a_agent import A2AAgent
from rasa.agents.schemas import AgentInput
from rasa.agents.schemas.agent_input import AgentInputSlot
from rasa.core.available_agents import (
    AgentConfig,
    AgentConfiguration,
    AgentInfo,
    ProtocolConfig,
)
from rasa.core.constants import (
    BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_INTERMEDIATE_MESSAGE,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY,
    BOT_UTTERANCE_AGENT_NAME_KEY,
    BOT_UTTERANCE_AGENT_TASK_ID_KEY,
    BOT_UTTERANCE_CONTEXT_ID_KEY,
    BOT_UTTERANCE_MESSAGE_ID_KEY,
    UTTER_SOURCE_METADATA_KEY,
)
from rasa.shared.core.events import BotUttered
from rasa.shared.exceptions import (
    AgentInitializationException,
    InvalidParameterException,
)


class StreamTracker:
    """Tracks async generator behavior for testing aclosing() fix."""

    def __init__(self) -> None:
        self.events_yielded = 0
        self.was_closed_properly = False

    def create_stream(
        self, event_to_yield: Any, max_yields: int = 3
    ) -> Callable[[], AsyncGenerator[Any, None]]:
        """Create an async generator that tracks behavior."""

        async def stream():
            try:
                for _ in range(max_yields):
                    self.events_yielded += 1
                    yield event_to_yield
            except GeneratorExit:
                # This indicates the generator was properly closed by aclosing()
                self.was_closed_properly = True
                raise

        return stream


@pytest.fixture
def stream_tracker() -> StreamTracker:
    """Fixture for tracking async generator behavior."""
    return StreamTracker()


@pytest.fixture
def immediate_create_task():
    """Patch asyncio.create_task in A2A module so done_callback fires immediately.

    Still schedules the real coroutine with the original create_task to preserve
    AsyncMock await counts on output channels.
    """
    import asyncio as _asyncio

    from rasa.agents.protocol.a2a import a2a_agent as a2a_mod

    orig_create_task = _asyncio.create_task

    def _immediate_task(coro):
        # schedule real coroutine to preserve awaited counts and behavior
        orig_create_task(coro)

        class _FakeDone:
            def exception(self):
                return None

        class _FakeTask:
            def add_done_callback(self, cb):
                cb(_FakeDone())

        return _FakeTask()

    with patch.object(a2a_mod.asyncio, "create_task", side_effect=_immediate_task):
        yield


@pytest.fixture(autouse=True)
def _mock_health_check_for_all_tests_except_dedicated(
    monkeypatch: MonkeyPatch, request: FixtureRequest
):
    # Skip mocking for dedicated health check tests
    if request.node.name.startswith("test_health_check_"):
        return

    async def _noop(self):
        return None

    monkeypatch.setattr(A2AAgent, "_perform_health_check", _noop)


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2ACardResolver")
async def test_a2a_agent_connect_with_local_agent_card(mock_card_resolver):
    # Create a mock configuration with a local agent card path
    config = MagicMock()
    config.configuration.agent_card = (
        "tests/agents/protocol/a2a/agent_cards/test_agent_card.json"
    )
    config.configuration.timeout = 30
    config.agent.name = "test_agent"
    config.agent.description = "A test agent"
    config.configuration.auth = None
    agent = A2AAgent.from_config(config)

    await agent.connect()
    assert agent is not None

    # Verify that the agent card was loaded from a file and not resolved from a URL
    mock_card_resolver.assert_not_called()


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2ACardResolver")
async def test_a2a_agent_connect_with_remote_agent_card(mock_card_resolver):
    # Create a mock configuration with a URL agent card path
    config = MagicMock()
    config.configuration.agent_card = "http://example.com/test_agent_card.json"
    config.configuration.timeout = 30
    config.agent.name = "test_agent"
    config.agent.description = "A test agent"
    config.configuration.auth = None
    agent = A2AAgent.from_config(config)

    # Mock the agent card resolver
    mock_card_resolver.return_value.get_agent_card = AsyncMock(
        return_value=A2AAgent._load_agent_card_from_file(
            "tests/agents/protocol/a2a/agent_cards/test_agent_card.json"
        )
    )
    await agent.connect()

    # Verify that the agent card was resolved from a URL
    mock_card_resolver.return_value.get_agent_card.assert_awaited_once()
    assert agent is not None


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2ACardResolver")
async def test_a2a_agent_connect_with_remote_agent_card_retries(mock_card_resolver):
    # Create a mock configuration with a URL agent card path
    config = MagicMock()
    config.configuration.agent_card = "http://example.com/test_agent_card.json"
    config.configuration.timeout = 1
    config.configuration.max_retries = 2
    config.agent.name = "test_agent"
    config.agent.description = "A test agent"
    agent = A2AAgent.from_config(config)

    # Mock the agent card resolver
    mock_card_resolver.return_value.get_agent_card.side_effect = A2AClientHTTPError(
        message="HTTP Error", status_code=500
    )

    with pytest.raises(AgentInitializationException):
        await agent.connect()

    # Verify that we retried the expected number of times
    assert mock_card_resolver.return_value.get_agent_card.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mock_response, expected_status",
    [
        (
            (
                Task(
                    context_id="abc",
                    id="abc-123",
                    status=TaskStatus(state=TaskState.input_required),
                ),
                None,
            ),
            AgentStatus.INPUT_REQUIRED,
        ),
        (
            (
                Task(
                    context_id="abc",
                    id="abc-123",
                    status=TaskStatus(state=TaskState.completed),
                    artifacts=[Artifact(artifact_id="artifact-1", parts=[])],
                ),
                None,
            ),
            AgentStatus.COMPLETED,
        ),
        (
            (
                Task(
                    context_id="abc",
                    id="abc-123",
                    status=TaskStatus(state=TaskState.failed),
                ),
                None,
            ),
            AgentStatus.RECOVERABLE_ERROR,
        ),
    ],
)
@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_streaming_agent(
    mock_init_client: MagicMock, mock_response, expected_status
):
    async def mock_stream_generator():
        yield mock_response

    mock_client = MagicMock()
    mock_client.send_message.return_value = mock_stream_generator()

    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    # Run the agent
    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    mock_client.send_message.assert_called_once()

    # Verify the output status
    assert output.status == expected_status


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_pooling_agent_triggers_on_non_terminal_task(mock_init_client):
    # Prepare a non-terminal task (working) and a terminal task (completed)

    # First response: non-terminal state (working)
    non_terminal_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.working),
    )
    # Final response: terminal (completed)
    completed_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="artifact-1", parts=[])],
    )

    # Mock send_message to yield a single tuple with non-terminal task
    async def mock_stream_generator():
        yield non_terminal_task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = mock_stream_generator()
    # Mock get_task: the first two calls returns non-terminal Task,
    # third call returns completed Task
    mock_client.get_task = AsyncMock(
        side_effect=[non_terminal_task, non_terminal_task, completed_task]
    )
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    # Run the agent
    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    # send_message should be called once
    mock_client.send_message.assert_called_once()
    # get_task should be called three times (2 non-terminal + 1 terminal)
    assert mock_client.get_task.call_count == 3
    # Final output should be COMPLETED
    assert output.status == AgentStatus.COMPLETED


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_pooling_sends_intermediate_messages(mock_init_client):
    # First response: non-terminal state (submitted) with message
    submitted_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(
            state=TaskState.submitted,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Submitted message"))],
                message_id="m2",
            ),
        ),
    )
    # Working response with message
    working_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Working message"))],
                message_id="m3",
            ),
        ),
    )

    # Final response: terminal (completed)
    completed_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="artifact-1", parts=[])],
    )

    # Mock send_message to yield a single tuple with non-terminal task
    async def mock_stream_generator():
        yield submitted_task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = mock_stream_generator()
    # Mock get_task: working task returned multiple times before completion
    mock_client.get_task = AsyncMock(
        side_effect=[
            working_task,
            working_task,
            working_task,
            working_task,
            completed_task,
        ]
    )
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    # Prepare a mock output channel and agent input with recipient id
    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = AsyncMock()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-123",
        ),
        output_channel=mock_output_channel,
    )

    # send_message should be called once
    mock_client.send_message.assert_called_once()
    # get_task should be called three times (4 non-terminal + 1 terminal)
    assert mock_client.get_task.call_count == 5
    # Final output should be COMPLETED
    assert output.status == AgentStatus.COMPLETED

    # Allow background tasks created via asyncio.create_task to run
    await asyncio.sleep(0)

    calls = mock_output_channel.send_text_message.await_args_list
    # Validate that the intermediate messages were sent only once each
    # despite repeated working status
    assert len(calls) == 2
    assert calls[0].kwargs["text"] == "Submitted message"
    assert calls[1].kwargs["text"] == "Working message"

    # Validate that the recipient_id was passed correctly each time
    assert calls[0].kwargs["recipient_id"] == "user-123"
    assert calls[1].kwargs["recipient_id"] == "user-123"

    assert output.events is not None
    bot_uttered_events = [
        event for event in output.events if isinstance(event, BotUttered)
    ]
    # Validate that the final output events contain the BotUttered events for each
    # intermediate message sent
    assert len(bot_uttered_events) == 2
    assert bot_uttered_events[0].text == "Submitted message"
    assert bot_uttered_events[1].text == "Working message"

    for event in bot_uttered_events:
        assert event.metadata is not None
        assert event.metadata.get(UTTER_SOURCE_METADATA_KEY) == "A2AAgent"
        assert event.metadata.get(BOT_UTTERANCE_AGENT_NAME_KEY) == "test_agent"
        assert event.metadata.get(BOT_UTTERANCE_AGENT_TASK_ID_KEY) == "abc-123"
        assert event.metadata.get(BOT_UTTERANCE_CONTEXT_ID_KEY) == "abc"

    assert bot_uttered_events[0].metadata.get(BOT_UTTERANCE_MESSAGE_ID_KEY) == "m2"
    assert bot_uttered_events[1].metadata.get(BOT_UTTERANCE_MESSAGE_ID_KEY) == "m3"


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_streaming_duplicate_intermediate_messages_not_deduped(
    mock_init_client: MagicMock, immediate_create_task
):
    # Streaming yields the same working message twice, then completes
    working_msg_1 = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Same message"))],
                message_id="m1",
            ),
        ),
    )
    working_msg_2 = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Same message"))],
                message_id="m2",
            ),
        ),
    )
    completed_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="a1", parts=[])],
    )

    async def stream():
        yield working_msg_1, None
        yield working_msg_2, None
        yield completed_task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = AsyncMock()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-123",
        ),
        output_channel=mock_output_channel,
    )

    await asyncio.sleep(0)

    # Since streaming should not dedup, both identical messages are sent
    calls = mock_output_channel.send_text_message.await_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["text"] == "Same message"
    assert calls[1].kwargs["text"] == "Same message"

    # And two BotUttered events recorded
    assert output.events is not None
    bot_uttered_events = [
        event for event in output.events if isinstance(event, BotUttered)
    ]
    assert len(bot_uttered_events) == 2
    assert bot_uttered_events[0].text == "Same message"
    assert bot_uttered_events[1].text == "Same message"
    # Metadata assertions
    ids = {e.metadata.get(BOT_UTTERANCE_MESSAGE_ID_KEY) for e in bot_uttered_events}
    assert ids == {"m1", "m2"}
    for event in bot_uttered_events:
        assert event.metadata.get(UTTER_SOURCE_METADATA_KEY) == "A2AAgent"
        assert (
            event.metadata.get(BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY)
            == BOT_UTTERANCE_AGENT_MESSAGE_TYPE_INTERMEDIATE_MESSAGE
        )
        assert event.metadata.get(BOT_UTTERANCE_AGENT_NAME_KEY) == "test_agent"
        assert event.metadata.get(BOT_UTTERANCE_AGENT_TASK_ID_KEY) == "abc-123"
        assert event.metadata.get(BOT_UTTERANCE_CONTEXT_ID_KEY) == "abc"


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_pooling_agent_handles_unknown_task_state(mock_init_client):
    # Prepare an unknown task state first, then a terminal task (completed)

    # First response: non-terminal state (unknown)
    non_terminal_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.unknown),
    )
    # Final response: terminal (completed)
    completed_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="artifact-1", parts=[])],
    )

    # Mock send_message to yield a single tuple with unknown task
    async def mock_stream_generator():
        yield non_terminal_task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = mock_stream_generator()
    # Mock get_task: the first two calls return unknown Task,
    # third call returns completed Task
    mock_client.get_task = AsyncMock(
        side_effect=[non_terminal_task, non_terminal_task, completed_task]
    )
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    # Run the agent with structlog capture
    with capture_logs() as logs:
        output = await agent.run(
            AgentInput(
                id="abc",
                metadata={},
                user_message="Test message",
                slots=[],
                conversation_history="",
                events=[],
            )
        )

    # send_message should be called once
    mock_client.send_message.assert_called_once()
    # get_task should be called three times (2 unknown + 1 terminal)
    assert mock_client.get_task.call_count == 3
    # Final output should be COMPLETED
    assert output.status == AgentStatus.COMPLETED

    # Ensure a warning about unknown task state was logged at least once
    assert any(
        log.get("event") == "a2a_agent.run_streaming_agent.unknown_task_state"
        for log in logs
    )


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_pooling_dedups_submitted_messages(mock_init_client: MagicMock):
    # Initial streaming yields non-terminal without message to enter polling
    first_non_terminal = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.working, message=None),
    )
    submitted_with_msg = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(
            state=TaskState.submitted,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Submitted message"))],
                message_id="m1",
            ),
        ),
    )
    completed_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="artifact-1", parts=[])],
    )

    async def stream():
        yield first_non_terminal, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    # Polling returns the same submitted message twice before completion
    mock_client.get_task = AsyncMock(
        side_effect=[submitted_with_msg, submitted_with_msg, completed_task]
    )
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = AsyncMock()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-123",
        ),
        output_channel=mock_output_channel,
    )

    await asyncio.sleep(0)

    # Only one submitted message should be sent despite duplicate polling responses
    calls = mock_output_channel.send_text_message.await_args_list
    assert len(calls) == 1
    assert calls[0].kwargs["text"] == "Submitted message"
    assert calls[0].kwargs["recipient_id"] == "user-123"

    # And only one BotUttered for that message
    assert output.events is not None
    bot_uttered_events = [
        event for event in output.events if isinstance(event, BotUttered)
    ]
    assert len(bot_uttered_events) == 1
    assert bot_uttered_events[0].text == "Submitted message"
    # Final status is COMPLETED
    assert output.status == AgentStatus.COMPLETED
    # For intermediate messages from polling, context id should
    # reflect the task's context
    assert bot_uttered_events[0].metadata.get(BOT_UTTERANCE_CONTEXT_ID_KEY) == "abc"


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_pooling_preserves_events_on_client_error_after_intermediate(
    mock_init_client: MagicMock,
):
    # Enter polling, then send one working message and fail
    first_non_terminal = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.working, message=None),
    )
    working_with_msg = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Working polling message"))],
                message_id="m2",
            ),
        ),
    )

    async def stream():
        yield first_non_terminal, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_client.get_task = AsyncMock(
        side_effect=[working_with_msg, A2AClientError("polling error")]
    )
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = AsyncMock()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-123",
        ),
        output_channel=mock_output_channel,
    )

    await asyncio.sleep(0)

    # One intermediate message was sent before client error
    calls = mock_output_channel.send_text_message.await_args_list
    assert len(calls) == 1
    assert calls[0].kwargs["text"] == "Working polling message"
    assert calls[0].kwargs["recipient_id"] == "user-123"

    # Output fatal error with preserved BotUttered
    assert output.status == AgentStatus.FATAL_ERROR
    assert output.events is not None
    bot_uttered_events = [
        event for event in output.events if isinstance(event, BotUttered)
    ]
    assert len(bot_uttered_events) == 1
    assert bot_uttered_events[0].text == "Working polling message"
    # Metadata assertions
    events = bot_uttered_events[0]
    assert events.metadata.get(UTTER_SOURCE_METADATA_KEY) == "A2AAgent"
    assert (
        events.metadata.get(BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY)
        == BOT_UTTERANCE_AGENT_MESSAGE_TYPE_INTERMEDIATE_MESSAGE
    )
    assert events.metadata.get(BOT_UTTERANCE_AGENT_NAME_KEY) == "test_agent"
    assert events.metadata.get(BOT_UTTERANCE_AGENT_TASK_ID_KEY) == "abc-123"
    assert events.metadata.get(BOT_UTTERANCE_MESSAGE_ID_KEY) == "m2"


def test_handle_task_returns_none_for_unknown_state():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )

    task = Task(
        context_id="ctx",
        id="t1",
        status=TaskStatus(state=TaskState.unknown),
    )
    agent_input = AgentInput(
        id="agent-x",
        metadata={},
        user_message="",
        slots=[],
        conversation_history="",
        events=[],
    )

    result = agent._handle_task(agent_input=agent_input, task=task, generated_events=[])
    assert result is None


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_streaming_agent_handles_unknown_then_completed_no_poll(
    mock_init_client: MagicMock,
):
    # Streaming yields unknown first, then completed;
    # ensure no polling and warning logged

    unknown_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.unknown),
    )
    completed_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="artifact-1", parts=[])],
    )

    async def stream():
        yield unknown_task, None
        yield completed_task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_client.get_task = AsyncMock()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    with capture_logs() as logs:
        output = await agent.run(
            AgentInput(
                id="abc",
                metadata={},
                user_message="Test message",
                slots=[],
                conversation_history="",
                events=[],
            )
        )

    mock_client.send_message.assert_called_once()
    assert mock_client.get_task.call_count == 0
    assert output.status == AgentStatus.COMPLETED

    assert any(
        log.get("event") == "a2a_agent.run_streaming_agent.unknown_task_state"
        for log in logs
    )


@pytest.mark.asyncio
async def test_run_client_not_initialized_returns_fatal_error():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )

    # Do NOT call connect(); ensure client is uninitialized
    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.FATAL_ERROR
    assert output.error_message == "Client not initialized"


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_no_events_returns_recoverable_error(mock_init_client: MagicMock):
    async def empty_stream():
        if False:
            yield None  # pragma: no cover

    mock_client = MagicMock()
    mock_client.send_message.return_value = empty_stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_card = MagicMock()
        mock_card.url = "http://example.com"
        mock_load_card.return_value = mock_card
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.RECOVERABLE_ERROR
    assert "No events received" in (output.error_message or "")


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_message_response_sets_input_required_and_metadata(
    mock_init_client: MagicMock,
):
    response_message = Message(
        role=Role.user,
        parts=[
            Part(root=TextPart(text="How can I help you?")),
        ],
        message_id="m1",
        context_id="ctx-1",
        task_id=None,
    )

    async def stream():
        yield response_message

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_card = MagicMock()
        mock_card.url = "http://example.com"
        mock_load_card.return_value = mock_card
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Hello!",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.INPUT_REQUIRED
    assert output.response_message == "How can I help you?"
    assert output.metadata.get(A2A_AGENT_CONTEXT_ID_KEY) == "ctx-1"


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_unexpected_response_type_returns_fatal_error(
    mock_init_client: MagicMock,
):
    async def stream():
        yield 42  # unexpected type

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_card = MagicMock()
        mock_card.url = "http://example.com"
        mock_load_card.return_value = mock_card
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.FATAL_ERROR
    assert "Unexpected response type" in (output.error_message or "")


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_polling_timeout_returns_fatal_error(
    mock_init_client: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    non_terminal_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.working),
    )

    async def stream():
        yield non_terminal_task, None

    async def slow_get_task(*args, **kwargs):
        # Simulate slow server response exceeding max_wait
        await asyncio.sleep(2.0)
        return non_terminal_task

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_client.get_task = slow_get_task
    mock_init_client.return_value = mock_client

    # Set only the overall timeout to 1s
    import rasa.agents.protocol.a2a.a2a_agent as a2a_mod

    monkeypatch.setattr(a2a_mod, "A2A_TASK_POOLING_MAX_WAIT", 1.0, raising=False)

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_card = MagicMock()
        mock_card.url = "http://example.com"
        mock_load_card.return_value = mock_card
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.FATAL_ERROR
    assert "Polling timed out" in (output.error_message or "")


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_polling_error_returns_fatal_error(mock_init_client: MagicMock):
    non_terminal_task = Task(
        context_id="abc",
        id="abc-123",
        status=TaskStatus(state=TaskState.working),
    )

    async def stream():
        yield non_terminal_task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_client.get_task = AsyncMock(side_effect=A2AClientError("error"))
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_card = MagicMock()
        mock_card.url = "http://example.com"
        mock_load_card.return_value = mock_card
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.FATAL_ERROR
    assert "Polling error" in (output.error_message or "")


def test_from_config_missing_agent_card_raises():
    with pytest.raises(InvalidParameterException):
        A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="A test agent",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(agent_card=None),
            )
        )


def test_from_config_defaults_timeout_and_retries():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(
                agent_card="some/path", timeout=None, max_retries=None
            ),
        )
    )

    assert agent._timeout == AGENT_DEFAULT_TIMEOUT_SECONDS
    assert agent._max_retries == AGENT_DEFAULT_MAX_RETRIES


def test_from_config_respects_provided_timeout_and_retries():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(
                agent_card="some/path", timeout=7, max_retries=9
            ),
        )
    )

    assert agent._timeout == 7
    assert agent._max_retries == 9


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_connect_client_init_failure_raises_agent_init_error(
    mock_init_client: MagicMock,
):
    mock_init_client.side_effect = RuntimeError("boom")

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(
                agent_card="tests/agents/protocol/a2a/agent_cards/test_agent_card.json"
            ),
        )
    )

    with pytest.raises(AgentInitializationException):
        await agent.connect()


def test_prepare_message_uses_history_without_context_id():
    # No context/task id in metadata: should send full conversation history
    agent_input = AgentInput(
        id="agent-a",
        user_message="ignored",
        slots=[],
        conversation_history="User: Hi\nBot: Hello",
        events=[],
        metadata={},
    )

    message = A2AAgent._prepare_message(agent_input)
    # one TextPart with conversation history
    assert len(message.parts) == 1
    assert isinstance(message.parts[0].root, TextPart)
    assert message.parts[0].root.text == "User: Hi\nBot: Hello"
    assert message.context_id is None
    assert message.task_id is None


def test_prepare_message_uses_last_user_message_with_context_and_slots():
    agent_input = AgentInput(
        id="agent-a",
        user_message="Latest user message",
        slots=[
            AgentInputSlot(name="city", value="Berlin", type="text"),
            AgentInputSlot(
                name="temperature", value=None, type="text"
            ),  # should be filtered out
        ],
        conversation_history="User: Hi\nBot: Hello",
        events=[],
        metadata={
            A2A_AGENT_CONTEXT_ID_KEY: "ctx-123",
            A2A_AGENT_TASK_ID_KEY: "task-999",
        },
    )

    message = A2AAgent._prepare_message(agent_input)
    # Expect 2 parts: last user message + slots DataPart
    assert len(message.parts) == 2
    assert isinstance(message.parts[0].root, TextPart)
    assert message.parts[0].root.text == "Latest user message"
    assert isinstance(message.parts[1].root, DataPart)
    # Ensure only non-None slot is included
    slots_payload = message.parts[1].root.data
    assert slots_payload == {"slots": {"city": "Berlin"}}
    assert message.context_id == "ctx-123"
    assert message.task_id == "task-999"


def test_generate_response_message_from_parts_text_and_file_uri():
    parts = [
        Part(root=TextPart(text="Line 1")),
        Part(
            root=FilePart(
                file=FileWithUri(
                    uri="https://file", name="f.txt", mime_type="text/plain"
                )
            )
        ),
        Part(root=DataPart(data={"key": "value"})),  # ignored in response message
    ]
    result = A2AAgent._generate_response_message_from_parts(parts)
    # Text + file URI lines, no DataPart
    assert result == "Line 1\nFile: https://file"


@pytest.mark.parametrize(
    "parts1, parts2, expected_result",
    [
        (
            [Part(root=TextPart(text="Line 1"))],
            [Part(root=TextPart(text="Line 2"))],
            "Line 1\nLine 2",
        ),
        (
            [Part(root=TextPart(text="duplicate"))],
            [Part(root=TextPart(text="duplicate"))],
            "duplicate",
        ),
        (
            [Part(root=TextPart(text="Line 1")), Part(root=TextPart(text="Line 2"))],
            [Part(root=TextPart(text="Line 3"))],
            "Line 1\nLine 2\nLine 3",
        ),
    ],
)
def test_generate_completed_response_message_merges_status_and_artifacts(
    parts1: List[Part], parts2: List[Part], expected_result: str
):
    # Status message parts + artifact parts merged
    status_msg = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text="Done"))],
        message_id="m",
        context_id=None,
        task_id=None,
    )
    artifacts = [
        Artifact(artifact_id="a1", parts=parts1),
        Artifact(artifact_id="a2", parts=parts2),
    ]
    task = Task(
        context_id="ctx",
        id="t",
        status=TaskStatus(state=TaskState.completed, message=status_msg),
        artifacts=artifacts,
    )

    result = A2AAgent._generate_completed_response_message(task)
    # Preserves order: status message then artifacts
    assert result == f"Done\n{expected_result}"


def test_generate_structured_results_from_artifacts_accumulates_previous():
    artifacts = [
        Artifact(
            artifact_id="a1",
            parts=[
                Part(root=DataPart(data={"d": 1})),
                Part(
                    root=FilePart(file=FileWithUri(uri="u1", name="n1", mime_type="t1"))
                ),
            ],
        ),
        Artifact(
            artifact_id="a2",
            parts=[Part(root=TextPart(text="ignored"))],
        ),
    ]

    # Existing tool results in metadata
    prev = [[{"name": "prev_0_0", "type": "data", "result": {"p": 1}}]]
    agent_input = AgentInput(
        id="agent-x",
        user_message="",
        slots=[],
        conversation_history="",
        events=[],
        metadata={AGENT_METADATA_STRUCTURED_RESULTS_KEY: prev},
    )

    results = A2AAgent._generate_structured_results_from_artifacts(
        agent_input, artifacts
    )
    # Should append a new list for current iteration with two results
    assert len(results) == 2
    assert results[0] == prev[0]
    current = results[1]
    assert len(current) == 2
    # Names encoded with agent id and indices
    assert current[0]["name"] == "agent-x_0_0"
    assert current[0]["type"] == "data"
    assert current[0]["result"] == {"d": 1}
    assert current[1]["name"] == "agent-x_0_1"
    assert current[1]["type"] == "file"
    assert current[1]["result "] == {"uri": "u1", "name": "n1", "mime_type": "t1"}


@pytest.mark.asyncio
async def test_load_agent_card_file_not_found_raises_agent_init_error():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(
                agent_card="tests/agents/protocol/a2a/agent_cards/does_not_exist.json"
            ),
        )
    )

    with pytest.raises(AgentInitializationException):
        await agent.connect()


@pytest.mark.asyncio
async def test_health_check_uninitialized_raises_agent_init_error():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )

    with pytest.raises(AgentInitializationException):
        await agent._perform_health_check()


@pytest.mark.asyncio
async def test_health_check_succeeds_on_message_event():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    agent.agent_card = MagicMock()

    response_message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text="ok"))],
        message_id="m1",
        context_id=None,
        task_id=None,
    )

    async def stream():
        yield response_message

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    agent._client = mock_client

    await agent._perform_health_check()
    mock_client.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_health_check_succeeds_on_task_event():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    agent.agent_card = MagicMock()

    task = Task(
        context_id="ctx",
        id="t1",
        status=TaskStatus(state=TaskState.working),
    )

    async def stream():
        yield task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    agent._client = mock_client

    await agent._perform_health_check()
    mock_client.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_health_check_unexpected_event_type_raises_agent_init_error():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    agent.agent_card = MagicMock()

    async def stream():
        yield 123  # unexpected type

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    agent._client = mock_client

    with pytest.raises(AgentInitializationException) as exc:
        await agent._perform_health_check()
    assert "Unexpected response type" in str(exc.value)


@pytest.mark.asyncio
async def test_health_check_exception_wrapped_as_agent_init_error():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    agent.agent_card = MagicMock()

    mock_client = MagicMock()
    mock_client.send_message.side_effect = RuntimeError("error")
    agent._client = mock_client

    with pytest.raises(AgentInitializationException) as exc:
        await agent._perform_health_check()
    assert "Health check failed for A2A agent 'test_agent'" in str(exc.value)


@pytest.mark.asyncio
async def test_health_check_no_events_raises_agent_init_error():
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    agent.agent_card = MagicMock()

    async def empty_stream():
        if False:
            yield None  # pragma: no cover

    mock_client = MagicMock()
    mock_client.send_message.return_value = empty_stream()
    agent._client = mock_client

    with pytest.raises(AgentInitializationException) as exc:
        await agent._perform_health_check()
    assert "no events received" in str(exc.value)


@pytest.mark.asyncio
async def test_health_check_properly_closes_async_generator(
    stream_tracker: StreamTracker,
) -> None:
    """Test that health check properly closes async generator."""
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    agent.agent_card = MagicMock()

    response_message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text="ok"))],
        message_id="m1",
        context_id=None,
        task_id=None,
    )

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream_tracker.create_stream(
        response_message
    )()
    agent._client = mock_client

    # This should succeed and properly close the generator
    await agent._perform_health_check()

    # Verify the generator was properly closed by aclosing()
    assert (
        stream_tracker.was_closed_properly
    ), "Generator should be properly closed by aclosing()"
    assert (
        stream_tracker.events_yielded == 1
    ), "Only the first event should be yielded before generator is closed"
    mock_client.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_run_method_properly_closes_async_generator(
    stream_tracker: StreamTracker,
) -> None:
    """Test run method properly closes async generator."""
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    agent.agent_card = MagicMock()

    # Create a task that will be returned immediately
    task = Task(
        context_id="ctx",
        id="t1",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="artifact-1", parts=[])],
    )

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream_tracker.create_stream((task, None))()
    agent._client = mock_client

    # This should succeed and properly close the generator
    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    # Verify the generator was properly closed by aclosing()
    assert (
        stream_tracker.was_closed_properly
    ), "Generator should be properly closed by aclosing()"
    assert (
        stream_tracker.events_yielded == 1
    ), "Only the first event should be yielded before generator is closed"
    assert output.status == AgentStatus.COMPLETED
    mock_client.send_message.assert_called_once()


class TestA2AAgentAuthIntegration:
    """Test cases for A2A agent authentication integration."""

    @pytest.mark.asyncio
    @patch("rasa.agents.protocol.a2a.a2a_agent.ClientFactory")
    async def test_init_client_with_none_auth_config(self, mock_client_factory):
        """Test that _init_client works correctly when auth config is None."""
        # Mock the client factory
        mock_client = MagicMock()
        mock_client_factory.return_value.create.return_value = mock_client

        agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="A test agent",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(agent_card="some/path", auth=None),
            )
        )

        # Call _init_client
        result = agent._init_client()

        # Verify client factory was called with auth=None
        mock_client_factory.assert_called_once()
        call_args = mock_client_factory.call_args
        client_config = call_args[1]["config"]
        assert client_config.httpx_client.auth is None

        assert result == mock_client

    @pytest.mark.asyncio
    @patch("rasa.agents.protocol.a2a.a2a_agent.ClientFactory")
    async def test_init_client_with_bearer_token_auth(self, mock_client_factory):
        """Test that _init_client works correctly with bearer token authentication."""
        # Mock the client factory
        mock_client = MagicMock()
        mock_client_factory.return_value.create.return_value = mock_client

        auth_config = {"token": "bearer_token_123"}

        agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="A test agent",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(
                    agent_card="some/path", auth=auth_config
                ),
            )
        )

        # Call _init_client
        result = agent._init_client()

        # Verify client factory was called with auth strategy
        mock_client_factory.assert_called_once()
        call_args = mock_client_factory.call_args
        client_config = call_args[1]["config"]
        # The auth strategy should be a real auth object (not None)
        assert client_config.httpx_client.auth is not None

        assert result == mock_client

    @pytest.mark.asyncio
    @patch("rasa.agents.protocol.a2a.a2a_agent.ClientFactory")
    async def test_init_client_with_api_key_auth(self, mock_client_factory):
        """Test that _init_client works correctly with API key authentication."""
        # Mock the client factory
        mock_client = MagicMock()
        mock_client_factory.return_value.create.return_value = mock_client

        auth_config = {"api_key": "api_key_123", "header_name": "X-API-Key"}

        agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="A test agent",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(
                    agent_card="some/path", auth=auth_config
                ),
            )
        )

        # Call _init_client
        result = agent._init_client()

        # Verify client factory was called with auth strategy
        mock_client_factory.assert_called_once()
        call_args = mock_client_factory.call_args
        client_config = call_args[1]["config"]
        # The auth strategy should be a real auth object (not None)
        assert client_config.httpx_client.auth is not None

        assert result == mock_client

    @pytest.mark.asyncio
    @patch("rasa.agents.protocol.a2a.a2a_agent.ClientFactory")
    async def test_init_client_with_oauth2_auth(self, mock_client_factory):
        """Test that _init_client works correctly with OAuth2 authentication."""
        # Mock the client factory
        mock_client = MagicMock()
        mock_client_factory.return_value.create.return_value = mock_client

        auth_config = {
            "oauth": {
                "client_id": "client_id_123",
                "client_secret": "client_secret_456",
                "token_url": "https://auth.example.com/token",
                "scope": "read write",
            }
        }

        agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="A test agent",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(
                    agent_card="some/path", auth=auth_config
                ),
            )
        )

        # Call _init_client
        result = agent._init_client()

        # Verify client factory was called with auth strategy
        mock_client_factory.assert_called_once()
        call_args = mock_client_factory.call_args
        client_config = call_args[1]["config"]
        # The auth strategy should be a real auth object (not None)
        assert client_config.httpx_client.auth is not None

        assert result == mock_client

    @pytest.mark.asyncio
    async def test_init_client_with_invalid_auth_config(self):
        """Test that _init_client handles invalid auth configuration gracefully."""
        from rasa.shared.exceptions import AgentAuthInitializationException

        invalid_auth_config = {"invalid_key": "invalid_value"}

        agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="A test agent",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(
                    agent_card="some/path", auth=invalid_auth_config
                ),
            )
        )

        # Call _init_client should raise the auth exception
        with pytest.raises(AgentAuthInitializationException):
            agent._init_client()

        # Call connect should raise the initialization exception
        with pytest.raises(AgentInitializationException):
            await agent.connect()

    @pytest.mark.asyncio
    @patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
    async def test_connect_with_auth_config(self, mock_init_client):
        """Test that connect() properly initializes client with auth configuration."""
        # Mock client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client

        auth_config = {"token": "test_token"}

        agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="A test agent",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(
                    agent_card="tests/agents/protocol/a2a/agent_cards/test_agent_card.json",
                    auth=auth_config,
                ),
            )
        )

        # Call connect
        await agent.connect()

        # Verify _init_client was called (which should use the auth config)
        mock_init_client.assert_called_once()

    @pytest.mark.asyncio
    @patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
    async def test_connect_without_auth_config(self, mock_init_client):
        """Test that connect() works correctly without auth configuration."""
        # Mock client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client

        agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="A test agent",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(
                    agent_card="tests/agents/protocol/a2a/agent_cards/test_agent_card.json",
                    auth=None,
                ),
            )
        )

        # Call connect
        await agent.connect()

        # Verify _init_client was called
        mock_init_client.assert_called_once()


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_jsonrpc_error_internal_is_recoverable(
    mock_init_client: MagicMock,
):
    async def raising_stream():
        if False:
            yield None  # pragma: no cover

        # Use real error classes; ensure `.error` attr exists for agent handling
        class _JSONRPCError(A2AClientJSONRPCError):
            def __init__(self) -> None:
                self.error = InternalError(message="internal")

        raise _JSONRPCError()

    mock_client = MagicMock()
    mock_client.send_message.return_value = raising_stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.RECOVERABLE_ERROR
    assert "internal" in output.error_message


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_jsonrpc_error_other_is_fatal(
    mock_init_client: MagicMock,
):
    async def raising_stream():
        if False:
            yield None  # pragma: no cover

        # Use real error class; set `.error` to a non-internal type
        class _JSONRPCError(A2AClientJSONRPCError):
            def __init__(self) -> None:
                self.error = ValueError("error")

        raise _JSONRPCError()

    mock_client = MagicMock()
    mock_client.send_message.return_value = raising_stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.FATAL_ERROR
    assert "error" in output.error_message


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_send_message_client_error_is_fatal(
    mock_init_client: MagicMock,
):
    async def raising_stream():
        if False:
            yield None  # pragma: no cover
        raise A2AClientError("oops")

    mock_client = MagicMock()
    mock_client.send_message.return_value = raising_stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.FATAL_ERROR
    assert "Send message error" in output.error_message


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_pooling_missing_task_id_returns_fatal_error(
    mock_init_client: MagicMock,
):
    async def stream():
        yield (
            Task(context_id="abc", id="", status=TaskStatus(state=TaskState.working)),
            None,
        )

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.FATAL_ERROR
    assert "Missing task_id for polling" in output.error_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        TaskState.canceled,
        TaskState.rejected,
        TaskState.auth_required,
    ],
)
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_run_streaming_agent_unsuccessful_terminal_states_are_recoverable(
    mock_init_client: MagicMock, state
):
    async def stream():
        yield Task(context_id="abc", id="abc-123", status=TaskStatus(state=state)), None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.RECOVERABLE_ERROR
    assert output.error_message.startswith("Task state: ")


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_task_metadata_propagation_input_required(mock_init_client: MagicMock):
    task = Task(
        context_id="ctx-42",
        id="task-42",
        status=TaskStatus(
            state=TaskState.input_required,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="need input"))],
                message_id="m",
                context_id=None,
                task_id=None,
            ),
        ),
    )

    async def stream():
        yield task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="hi",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.INPUT_REQUIRED
    assert output.metadata.get(A2A_AGENT_CONTEXT_ID_KEY) == "ctx-42"
    assert output.metadata.get(A2A_AGENT_TASK_ID_KEY) == "task-42"


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_task_metadata_propagation_completed(mock_init_client: MagicMock):
    task = Task(
        context_id="ctx-77",
        id="task-77",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="a1", parts=[])],
    )

    async def stream():
        yield task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="hi",
            slots=[],
            conversation_history="",
            events=[],
        )
    )

    assert output.status == AgentStatus.COMPLETED
    assert output.metadata.get(A2A_AGENT_CONTEXT_ID_KEY) == "ctx-77"
    assert output.metadata.get(A2A_AGENT_TASK_ID_KEY) == "task-77"


def test_structured_results_no_previous_creates_current_iteration():
    artifacts = [
        Artifact(
            artifact_id="a1",
            parts=[Part(root=DataPart(data={"k": 1}))],
        )
    ]

    agent_input = AgentInput(
        id="agent-y",
        user_message="",
        slots=[],
        conversation_history="",
        events=[],
        metadata={},
    )

    results = A2AAgent._generate_structured_results_from_artifacts(
        agent_input, artifacts
    )
    assert isinstance(results, list)
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0]["name"] == "agent-y_0_0"
    assert results[0][0]["type"] == "data"
    assert results[0][0]["result"] == {"k": 1}


def test_structured_results_only_text_parts_adds_empty_iteration():
    artifacts = [Artifact(artifact_id="a1", parts=[Part(root=TextPart(text="t1"))])]

    prev = [[{"name": "prev", "type": "data", "result": {"p": 1}}]]
    agent_input = AgentInput(
        id="agent-z",
        user_message="",
        slots=[],
        conversation_history="",
        events=[],
        metadata={AGENT_METADATA_STRUCTURED_RESULTS_KEY: prev},
    )

    results = A2AAgent._generate_structured_results_from_artifacts(
        agent_input, artifacts
    )
    assert len(results) == 2
    assert results[0] == prev[0]
    assert results[1] == []


def test_load_agent_card_invalid_json_raises_agent_init_error(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ invalid json }")

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card=str(bad_json)),
        )
    )

    with pytest.raises(AgentInitializationException):
        asyncio.get_event_loop().run_until_complete(agent.connect())


def test_load_agent_card_validation_error_raises_agent_init_error(tmp_path):
    invalid_card = tmp_path / "invalid_card.json"
    invalid_card.write_text('{"name": 123}')

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card=str(invalid_card)),
        )
    )

    with pytest.raises(AgentInitializationException):
        asyncio.get_event_loop().run_until_complete(agent.connect())


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
@pytest.mark.parametrize(
    "mock_final_task_event",
    # The intermediate messages should be sent regardless of the final task state
    [
        Task(
            context_id="ctx-1",
            id="task-1",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[Artifact(artifact_id="a1", parts=[])],
        ),
        Task(
            context_id="ctx-1",
            id="task-1",
            status=TaskStatus(state=TaskState.input_required),
            artifacts=[Artifact(artifact_id="a1", parts=[])],
        ),
        Task(
            context_id="ctx-1",
            id="task-1",
            status=TaskStatus(state=TaskState.failed),
            artifacts=[Artifact(artifact_id="a1", parts=[])],
        ),
    ],
)
async def test_intermediate_messages_sent_submitted_and_working_tasks_during_streaming(
    mock_init_client: MagicMock, mock_final_task_event: Task, immediate_create_task
):
    # Create a stream that yields:
    # 1) submitted task with message
    # 2) submitted task without message (no intermediate message should be sent)
    # 3) working task with message
    # 4) working task without message (no intermediate message should be sent)
    # 5) completed task
    submitted_task_with_msg = Task(
        context_id="ctx-1",
        id="task-1",
        status=TaskStatus(
            state=TaskState.submitted,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Submitted message"))],
                message_id="m1",
            ),
        ),
    )
    submitted_task_no_msg = Task(
        context_id="ctx-1",
        id="task-1",
        status=TaskStatus(state=TaskState.submitted, message=None),
    )
    working_with_msg = Task(
        context_id="ctx-1",
        id="task-1",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Working message"))],
                message_id="m2",
            ),
        ),
    )
    working_no_msg = Task(
        context_id="ctx-1",
        id="task-1",
        status=TaskStatus(state=TaskState.working, message=None),
    )

    async def stream():
        yield submitted_task_with_msg, None
        yield submitted_task_no_msg, None
        yield working_with_msg, None
        yield working_no_msg, None
        yield mock_final_task_event, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    # Prepare a mock output channel and agent input with recipient id
    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = AsyncMock()

    output = await agent.run(
        AgentInput(
            id="agent-1",
            metadata={},
            user_message="Hi",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-123",
        ),
        output_channel=mock_output_channel,
    )

    # Allow background tasks created via asyncio.create_task to run
    await asyncio.sleep(0)

    # Validate that only two non-empty messages were sent to the output channel
    calls = mock_output_channel.send_text_message.await_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["text"] == "Submitted message"
    assert calls[1].kwargs["text"] == "Working message"

    # Validate that the recipient_id was passed correctly each time
    assert calls[0].kwargs["recipient_id"] == "user-123"
    assert calls[1].kwargs["recipient_id"] == "user-123"

    # Validate that the final output events contain the BotUttered events for each
    # intermediate message sent
    assert output.events is not None
    bot_uttered_events = [
        event for event in output.events if isinstance(event, BotUttered)
    ]
    assert len(bot_uttered_events) == 2
    assert bot_uttered_events[0].text == "Submitted message"
    assert bot_uttered_events[1].text == "Working message"


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_intermediate_message_events_preserved_after_jsonrpc_error(
    mock_init_client: MagicMock, immediate_create_task
):
    working_with_msg = Task(
        context_id="ctx-1",
        id="task-1",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Working message"))],
                message_id="m2",
            ),
        ),
    )

    class _TestJSONRPCError(A2AClientJSONRPCError):
        def __init__(self) -> None:
            # Provide `.error` attribute as used by agent implementation
            self.error = InternalError(message="rpc failed")

    async def stream():
        # yield a working update with message, then raise JSON-RPC error
        yield working_with_msg, None
        raise _TestJSONRPCError()

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = AsyncMock()

    output = await agent.run(
        AgentInput(
            id="agent-1",
            metadata={},
            user_message="Hi",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-123",
        ),
        output_channel=mock_output_channel,
    )

    # Allow background tasks to complete
    await asyncio.sleep(0)

    # Intermediate working message should have been sent
    calls = mock_output_channel.send_text_message.await_args_list
    assert len(calls) == 1
    assert calls[0].kwargs["text"] == "Working message"
    assert calls[0].kwargs["recipient_id"] == "user-123"

    # Output should be recoverable error (due to InternalError) and include BotUttered
    assert output.status == AgentStatus.RECOVERABLE_ERROR
    assert output.events is not None
    bot_uttered_events = [
        event for event in output.events if isinstance(event, BotUttered)
    ]
    assert len(bot_uttered_events) == 1
    assert bot_uttered_events[0].text == "Working message"
    # Metadata assertions
    event = bot_uttered_events[0]
    assert event.metadata.get(UTTER_SOURCE_METADATA_KEY) == "A2AAgent"
    assert (
        event.metadata.get(BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY)
        == BOT_UTTERANCE_AGENT_MESSAGE_TYPE_INTERMEDIATE_MESSAGE
    )
    assert event.metadata.get(BOT_UTTERANCE_AGENT_NAME_KEY) == "test_agent"
    assert event.metadata.get(BOT_UTTERANCE_AGENT_TASK_ID_KEY) == "task-1"
    assert event.metadata.get(BOT_UTTERANCE_MESSAGE_ID_KEY) == "m2"


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_intermediate_message_events_preserved_after_client_error(
    mock_init_client: MagicMock, immediate_create_task
):
    working_with_msg = Task(
        context_id="ctx-2",
        id="task-2",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Still working"))],
                message_id="m3",
            ),
        ),
    )

    async def stream():
        yield working_with_msg, None
        raise A2AClientError("network down")

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = AsyncMock()

    output = await agent.run(
        AgentInput(
            id="agent-2",
            metadata={},
            user_message="Hello",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-999",
        ),
        output_channel=mock_output_channel,
    )

    await asyncio.sleep(0)

    calls = mock_output_channel.send_text_message.await_args_list
    assert len(calls) == 1
    assert calls[0].kwargs["text"] == "Still working"
    assert calls[0].kwargs["recipient_id"] == "user-999"

    # Client error should result in fatal error output, preserving BotUttered events
    assert output.status == AgentStatus.FATAL_ERROR
    assert output.events is not None
    bot_uttered_events = [
        event for event in output.events if isinstance(event, BotUttered)
    ]
    assert len(bot_uttered_events) == 1
    assert bot_uttered_events[0].text == "Still working"


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_streaming_intermediate_send_async_failure_does_not_stop_execution(
    mock_init_client: MagicMock,
):
    # Streaming: working with message then completed
    working_with_msg = Task(
        context_id="ctx-1",
        id="task-1",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Working message"))],
                message_id="m2",
            ),
        ),
    )
    completed_task = Task(
        context_id="ctx-1",
        id="task-1",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="a1", parts=[])],
    )

    async def stream():
        yield working_with_msg, None
        yield completed_task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    # Async send will raise when awaited inside the background task
    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = AsyncMock(
        side_effect=RuntimeError("send failed")
    )

    output = await agent.run(
        AgentInput(
            id="agent-1",
            metadata={},
            user_message="Hi",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-123",
        ),
        output_channel=mock_output_channel,
    )

    # Let the background task run and trigger the done_callback error handling
    await asyncio.sleep(0)

    # Although sending failed, execution should continue to completion
    assert output.status == AgentStatus.COMPLETED
    # Since sending failed, no BotUttered event should be recorded
    bot_uttered_events = [
        event for event in (output.events or []) if isinstance(event, BotUttered)
    ]
    assert len(bot_uttered_events) == 0
    # Attempted exactly once
    assert mock_output_channel.send_text_message.await_count == 1


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_streaming_intermediate_send_immediate_failure_does_not_stop_execution(
    mock_init_client: MagicMock,
):
    # Streaming: working with message then completed
    working_with_msg = Task(
        context_id="ctx-2",
        id="task-2",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Working message"))],
                message_id="m9",
            ),
        ),
    )
    completed_task = Task(
        context_id="ctx-2",
        id="task-2",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[Artifact(artifact_id="a2", parts=[])],
    )

    async def stream():
        yield working_with_msg, None
        yield completed_task, None

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    # Make send_text_message raise immediately (before a coroutine can be scheduled)
    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = MagicMock(
        side_effect=RuntimeError("immediate fail")
    )

    output = await agent.run(
        AgentInput(
            id="agent-2",
            metadata={},
            user_message="Hello",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-222",
        ),
        output_channel=mock_output_channel,
    )

    # Even though sending raised synchronously, execution should continue
    assert output.status == AgentStatus.COMPLETED
    # Since sending failed synchronously, no BotUttered event should be recorded
    bot_uttered_events = [
        event for event in (output.events or []) if isinstance(event, BotUttered)
    ]
    assert len(bot_uttered_events) == 0
    # Immediate call attempted once
    assert mock_output_channel.send_text_message.call_count == 1


def test_intermediate_metadata_omits_timestamp_when_absent() -> None:
    """Intermediate BotUttered metadata excludes timestamp if task has none."""
    task = Task(
        context_id="ctx-xyz",
        id="task-xyz",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="msg"))],
                message_id="m-xyz",
            ),
        ),
    )
    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    meta = agent._create_intermediate_message_bot_uttered_event_metadata(task)
    assert BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY not in meta


@pytest.mark.asyncio
@patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client")
async def test_intermediate_metadata_includes_timestamp_when_present(
    mock_init_client: MagicMock, immediate_create_task
) -> None:
    """Intermediate event should include timestamp when provided by task."""
    working_with_msg = Task(
        context_id="ctx-t",
        id="task-t",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="Timed msg"))],
                message_id="m-t",
            ),
            timestamp="2023-10-27T10:00:00Z",
        ),
    )

    async def stream():
        yield working_with_msg, None
        yield (
            Task(
                context_id="ctx-t",
                id="task-t",
                status=TaskStatus(state=TaskState.completed),
            ),
            None,
        )

    mock_client = MagicMock()
    mock_client.send_message.return_value = stream()
    mock_init_client.return_value = mock_client

    agent = A2AAgent.from_config(
        AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent",
                protocol=ProtocolConfig.A2A,
            ),
            configuration=AgentConfiguration(agent_card="some/path"),
        )
    )
    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
    ) as mock_load_card:
        mock_load_card.return_value = MagicMock()
        await agent.connect()

    mock_output_channel = MagicMock()
    mock_output_channel.send_text_message = AsyncMock()

    output = await agent.run(
        AgentInput(
            id="abc",
            metadata={},
            user_message="Test",
            slots=[],
            conversation_history="",
            events=[],
            recipient_id="user-1",
        ),
        output_channel=mock_output_channel,
    )

    await asyncio.sleep(0)
    assert output.events is not None
    bot_uttered_events = [
        e for e in output.events if isinstance(e, BotUttered) and e.text == "Timed msg"
    ]
    assert len(bot_uttered_events) == 1
    assert (
        bot_uttered_events[0].metadata.get(BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY)
        == "2023-10-27T10:00:00Z"
    )


def test_create_a2a_specific_metadata_from_message_only() -> None:
    base = {
        "keep": "x",
        BOT_UTTERANCE_MESSAGE_ID_KEY: "old-id",
        BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY: "oldts",
    }
    msg = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text="hello"))],
        message_id="m-msg",
        context_id="ctx-msg",
        task_id="task-msg",
    )
    result = A2AAgent._create_a2a_specific_metadata(base, message=msg)
    # base preserved (except stale keys removed)
    assert result["keep"] == "x"
    assert result[BOT_UTTERANCE_MESSAGE_ID_KEY] == "m-msg"
    assert result[A2A_AGENT_TASK_ID_KEY] == "task-msg"
    assert result[A2A_AGENT_CONTEXT_ID_KEY] == "ctx-msg"
    # stale timestamp removed since not provided by message/task
    assert BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY not in result


def test_create_a2a_specific_metadata_from_task_only() -> None:
    task = Task(
        context_id="ctx-task",
        id="task-1",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="hi"))],
                message_id="m-task",
            ),
            timestamp="2023-10-27T10:00:00Z",
        ),
    )
    result = A2AAgent._create_a2a_specific_metadata({}, task=task)
    assert result[A2A_AGENT_TASK_ID_KEY] == "task-1"
    assert result[A2A_AGENT_CONTEXT_ID_KEY] == "ctx-task"
    # message id from task.status.message
    assert result[BOT_UTTERANCE_MESSAGE_ID_KEY] == "m-task"
    assert result[BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY] == "2023-10-27T10:00:00Z"


def test_create_a2a_specific_metadata_precedence_and_stale_removal() -> None:
    base = {
        BOT_UTTERANCE_MESSAGE_ID_KEY: "old",
        BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY: 111.0,
    }
    message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text="m"))],
        message_id="m-msg",
        context_id="ctx-msg",
        task_id="task-msg",
    )
    task = Task(
        context_id="ctx-task",
        id="task-new",
        status=TaskStatus(
            state=TaskState.working,
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="from-status"))],
                message_id="m-status",
                context_id="ctx-status",
            ),
            timestamp="2023-10-27T10:00:00Z",
        ),
    )
    result = A2AAgent._create_a2a_specific_metadata(base, message=message, task=task)
    # Task id overrides message task_id
    assert result[A2A_AGENT_TASK_ID_KEY] == "task-new"
    # Status message context overrides task context and message context
    assert result[A2A_AGENT_CONTEXT_ID_KEY] == "ctx-status"
    # Status message id overrides message id (and removes stale base)
    assert result[BOT_UTTERANCE_MESSAGE_ID_KEY] == "m-status"
    # Timestamp present and not stale
    assert result[BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY] == "2023-10-27T10:00:00Z"


def test_create_a2a_specific_metadata_removes_stale_when_absent() -> None:
    base = {
        BOT_UTTERANCE_MESSAGE_ID_KEY: "old",
        BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY: "oldts",
    }
    message = Message(
        role=Role.user, parts=[Part(root=TextPart(text="x"))], message_id="new"
    )
    task = Task(
        context_id="ctx",
        id="t",
        status=TaskStatus(state=TaskState.working, message=None, timestamp=None),
    )
    result = A2AAgent._create_a2a_specific_metadata(base, message=message, task=task)
    assert result[BOT_UTTERANCE_MESSAGE_ID_KEY] == "new"
    assert BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY not in result
