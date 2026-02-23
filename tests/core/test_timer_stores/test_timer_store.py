from typing import Any, Dict, Optional

import pytest

from rasa.core.agent import Agent
from rasa.core.timer_managers.in_memory_timer_manager import InMemorySessionTimerManager
from rasa.core.timer_stores.timer_store import SessionTimer


def test_timer_manager_created_via_factory_in_agent():
    """Test that timer manager is created via factory when None is passed."""
    agent = Agent(timer_manager=None)

    assert agent.timer_manager is not None
    assert isinstance(agent.timer_manager, InMemorySessionTimerManager)


def test_session_timer_as_dict():
    """SessionTimer.as_dict serializes correctly."""
    timer = SessionTimer(
        sender_id="test_sender",
        session_id="test_session",
        scheduled_time=1234567890.0,
        metadata={"key": "value"},
    )

    result = timer.as_dict()

    assert result == {
        "sender_id": "test_sender",
        "session_id": "test_session",
        "scheduled_time": 1234567890.0,
        "metadata": {"key": "value"},
    }


@pytest.mark.parametrize(
    "data,expected_session_id,expected_metadata",
    [
        (
            {
                "sender_id": "test_sender",
                "session_id": "test_session",
                "scheduled_time": 1234567890.0,
                "metadata": {"key": "value"},
            },
            "test_session",
            {"key": "value"},
        ),
        (
            {
                "sender_id": "test_sender",
                "scheduled_time": 1234567890.0,
            },
            None,
            {},
        ),
    ],
)
def test_session_timer_from_dict(
    data: Dict[str, Any],
    expected_session_id: Optional[str],
    expected_metadata: Dict[str, Any],
):
    """SessionTimer.from_dict deserializes correctly with full and minimal data."""
    timer = SessionTimer.from_dict(data)

    assert timer.sender_id == "test_sender"
    assert timer.session_id == expected_session_id
    assert timer.scheduled_time == 1234567890.0
    assert timer.metadata == expected_metadata
