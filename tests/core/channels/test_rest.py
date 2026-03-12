"""Unit tests for the REST input channel."""

import asyncio
import time as time_mod
from unittest.mock import AsyncMock, MagicMock, patch

from a2a.types import Task, TaskState, TaskStatus

import rasa.core.run
from rasa.agents.core.cancellation import CancellationToken
from rasa.agents.core.types import AgentStatus
from rasa.agents.protocol.a2a.a2a_agent import A2AAgent
from rasa.agents.schemas import AgentInput
from rasa.core.available_agents import (
    AgentConfig,
    AgentConfiguration,
    AgentInfo,
    ProtocolConfig,
)
from rasa.core.channels.rest import RestInput
from rasa.core.processor import MessageProcessor


def _create_rest_app():
    """Create a Sanic app with the REST channel and a mock agent."""
    input_channel = RestInput()
    app = rasa.core.run.configure_app([input_channel], port=5004)

    mock_agent = MagicMock()
    app.ctx.agent = mock_agent
    return app, mock_agent


def test_cancel_background_tasks_returns_true_when_cancelled():
    app, mock_agent = _create_rest_app()
    mock_agent.cancel_background_tasks.return_value = True

    _, res = app.test_client.post("/webhooks/rest/cancel_background_tasks/sender-123")

    assert res.status_code == 200
    assert res.json == {"cancelled": True}
    mock_agent.cancel_background_tasks.assert_called_once_with("sender-123")


def test_cancel_background_tasks_returns_false_when_no_active_task():
    app, mock_agent = _create_rest_app()
    mock_agent.cancel_background_tasks.return_value = False

    _, res = app.test_client.post("/webhooks/rest/cancel_background_tasks/sender-456")

    assert res.status_code == 200
    assert res.json == {"cancelled": False}
    mock_agent.cancel_background_tasks.assert_called_once_with("sender-456")


def test_cancel_background_tasks_returns_500_on_exception():
    app, mock_agent = _create_rest_app()
    mock_agent.cancel_background_tasks.side_effect = RuntimeError("boom")

    _, res = app.test_client.post("/webhooks/rest/cancel_background_tasks/sender-789")

    assert res.status_code == 500
    assert res.json == {"cancelled": False}


def test_rest_cancel_endpoint_stops_a2a_polling():
    """POST /cancel_background_tasks/{sender_id} interrupts active A2A polling.

    Wires the REST endpoint through a real Agent.cancel_background_tasks →
    real processor token registry → real CancellationToken → real A2A
    polling loop.  Verifies the polling exits with CANCELLED.
    """
    import threading

    from rasa.core.agent import Agent

    # -- Processor with real token registry --
    processor = MagicMock(spec=MessageProcessor)
    processor._active_cancellation_tokens = {}
    processor.register_cancellation_token = (
        MessageProcessor.register_cancellation_token.__get__(processor)
    )
    processor.cancel_background_tasks = (
        MessageProcessor.cancel_background_tasks.__get__(processor)
    )

    # -- Agent with real cancel_background_tasks --
    agent = Agent.__new__(Agent)
    agent.processor = processor
    agent.tracker_store = MagicMock()
    agent.lock_store = MagicMock()

    # -- Sanic app wired to this agent --
    input_channel = RestInput()
    app = rasa.core.run.configure_app([input_channel], port=5004)
    app.ctx.agent = agent

    # -- A2A agent that polls forever --
    non_terminal_task = Task(
        context_id="ctx",
        id="ctx-001",
        status=TaskStatus(state=TaskState.working),
    )

    mock_client = MagicMock()
    mock_client.send_message.side_effect = lambda *a, **kw: _make_working_stream(
        non_terminal_task
    )
    mock_client.get_task = AsyncMock(return_value=non_terminal_task)

    loop = asyncio.new_event_loop()

    with patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client") as mock_init:
        mock_init.return_value = mock_client
        a2a_agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="Test",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(agent_card="some/path"),
            )
        )
        with patch(
            "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
        ) as mock_load:
            mock_card = MagicMock()
            mock_card.url = "http://example.com"
            mock_load.return_value = mock_card
            loop.run_until_complete(a2a_agent.connect())

    import rasa.agents.protocol.a2a.a2a_agent as a2a_mod

    original_max_wait = getattr(a2a_mod, "A2A_TASK_POLLING_MAX_WAIT", 60)

    sender_id = "rest-cancel-sender"
    token = CancellationToken()
    processor.register_cancellation_token(sender_id, token)

    # Run A2A polling in a background thread with its own event loop
    polling_result: dict = {}

    def _run_polling():
        try:
            a2a_mod.A2A_TASK_POLLING_MAX_WAIT = 120
            start = time_mod.monotonic()
            output = loop.run_until_complete(
                a2a_agent.run(
                    AgentInput(
                        id="ctx",
                        metadata={},
                        user_message="Test",
                        slots=[],
                        conversation_history="",
                        events=[],
                    ),
                    cancellation_token=token,
                )
            )
            polling_result["output"] = output
            polling_result["elapsed"] = time_mod.monotonic() - start
        finally:
            a2a_mod.A2A_TASK_POLLING_MAX_WAIT = original_max_wait

    polling_thread = threading.Thread(target=_run_polling)
    polling_thread.start()

    # Give polling time to start, then POST to the cancel endpoint
    import time

    time.sleep(0.3)

    _, res = app.test_client.post(f"/webhooks/rest/cancel_background_tasks/{sender_id}")
    assert res.status_code == 200
    assert res.json == {"cancelled": True}

    polling_thread.join(timeout=10)
    assert not polling_thread.is_alive(), "Polling thread should have finished"

    loop.close()

    output = polling_result["output"]
    elapsed = polling_result["elapsed"]

    assert output.status == AgentStatus.CANCELLED
    assert (output.metadata or {}).get("cancellation_reason") == "Polling cancelled"
    assert elapsed < 5.0, f"Polling should have exited promptly, took {elapsed:.2f}s"
    assert token.is_cancelled is True


async def _make_working_stream(task):
    yield (task, None)
