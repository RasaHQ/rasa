"""Unit tests for channels that implement conversation_blueprint."""

from typing import Any
from unittest.mock import Mock

import pytest
from sanic import Sanic

from rasa.core.agent import Agent
from rasa.core.channels.channel import InputChannel, register


# Test implementation
class DummyRuntimeAgentInputChannel(InputChannel):
    """Concrete implementation for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.received_agent: Any = None

    @classmethod
    def name(cls) -> str:
        return "dummy_conversation"

    def conversation_blueprint(self, agent: Any) -> Any:
        """Store the agent for testing."""
        from sanic import Blueprint

        self.received_agent = agent
        bp = Blueprint(self.name())
        return bp


@pytest.fixture
def channel() -> DummyRuntimeAgentInputChannel:
    """Create a channel for testing."""
    return DummyRuntimeAgentInputChannel()


@pytest.fixture
def mock_agent() -> Mock:
    """Create a mock Agent."""
    agent = Mock(spec=Agent)
    agent.__class__.__name__ = "Agent"
    return agent


def test_inherits_from_input_channel(channel: DummyRuntimeAgentInputChannel) -> None:
    """Test that the dummy runtime-agent channel is an input channel."""
    assert isinstance(channel, InputChannel)


def test_has_name_method(channel: DummyRuntimeAgentInputChannel) -> None:
    """Test that channel has the name() class method."""
    assert channel.name() == "dummy_conversation"


def test_url_prefix(channel: DummyRuntimeAgentInputChannel) -> None:
    """Test that url_prefix() works correctly."""
    assert channel.url_prefix() == "dummy_conversation"


def test_blueprint_receives_agent(
    channel: DummyRuntimeAgentInputChannel, mock_agent: Mock
) -> None:
    """Test that conversation_blueprint() receives the agent."""
    result = channel.conversation_blueprint(mock_agent)

    assert channel.received_agent is mock_agent
    assert result is not None  # Should return a blueprint


def test_register_passes_agent_that_resolves_after_registration(
    channel: DummyRuntimeAgentInputChannel, mock_agent: Mock
) -> None:
    """Test runtime-agent channels can access the agent loaded after registration."""
    app = Sanic("test_conversation_channel")

    register([channel], app, route="/webhooks/")
    app.ctx.agent = mock_agent

    assert channel.received_agent is not mock_agent
    assert channel.received_agent.model_metadata is mock_agent.model_metadata


def test_base_class_returns_no_conversation_blueprint() -> None:
    """Test that base InputChannel.conversation_blueprint() returns None."""

    class MinimalChannel(InputChannel):
        @classmethod
        def name(cls) -> str:
            return "minimal"

    channel = MinimalChannel()
    mock_agent = Mock(spec=Agent)

    assert channel.conversation_blueprint(mock_agent) is None


def test_register_raises_if_channel_provides_no_blueprint() -> None:
    """Test registration fails when neither blueprint hook is implemented."""

    class MinimalChannel(InputChannel):
        @classmethod
        def name(cls) -> str:
            return "minimal"

    app = Sanic("test_minimal_channel")

    with pytest.raises(NotImplementedError) as exc_info:
        register([MinimalChannel()], app, route="/webhooks/")

    assert "needs to provide blueprint() or conversation_blueprint()" in str(
        exc_info.value
    )
