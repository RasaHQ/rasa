"""Unit tests for the REST input channel."""

from unittest.mock import MagicMock

import rasa.core.run
from rasa.core.channels.rest import RestInput


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
