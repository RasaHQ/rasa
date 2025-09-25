import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from a2a.client import A2AClientError, A2AClientHTTPError
from a2a.types import (
    Artifact,
    DataPart,
    FilePart,
    FileWithUri,
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
from rasa.shared.exceptions import (
    AgentInitializationException,
    InvalidParameterException,
)


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

    result = agent._handle_task(agent_input=agent_input, task=task)
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
    assert slots_payload == {"slots": [{"name": "city", "value": "Berlin"}]}
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
    import rasa.agents.protocol.a2a.a2a_agent as a2a_mod

    class FakeInternalError(Exception):
        pass

    class FakeJSONRPCError(Exception):
        def __init__(self, error):
            super().__init__(str(error))
            self.error = error

    # Monkeypatch module-level symbols used by the agent
    a2a_mod.InternalError = FakeInternalError  # type: ignore[attr-defined]
    a2a_mod.A2AClientJSONRPCError = FakeJSONRPCError  # type: ignore[attr-defined]

    async def raising_stream():
        if False:
            yield None  # pragma: no cover
        raise FakeJSONRPCError(FakeInternalError("internal"))

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
    import rasa.agents.protocol.a2a.a2a_agent as a2a_mod

    class FakeJSONRPCError(Exception):
        def __init__(self, error):
            super().__init__(str(error))
            self.error = error

    a2a_mod.A2AClientJSONRPCError = FakeJSONRPCError  # type: ignore[attr-defined]

    async def raising_stream():
        if False:
            yield None  # pragma: no cover
        raise FakeJSONRPCError(ValueError("error"))

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
    import rasa.agents.protocol.a2a.a2a_agent as a2a_mod

    class FakeClientError(Exception):
        pass

    a2a_mod.A2AClientError = FakeClientError  # type: ignore[attr-defined]

    async def raising_stream():
        if False:
            yield None  # pragma: no cover
        raise FakeClientError("oops")

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
    from a2a.types import Task, TaskState, TaskStatus

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
    from a2a.types import Task, TaskStatus

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
    from a2a.types import Message, Part, Task, TaskState, TaskStatus, TextPart

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
    from a2a.types import Artifact, Task, TaskState, TaskStatus

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
    from a2a.types import Artifact, DataPart, Part

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
    from a2a.types import Artifact, Part, TextPart

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
