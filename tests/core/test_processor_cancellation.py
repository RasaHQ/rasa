"""Tests for the MessageProcessor cancellation token registry."""

from unittest.mock import MagicMock

from rasa.agents.core.cancellation import CancellationToken


def _create_processor_with_registry():
    """Create a minimal mock of MessageProcessor with the registry methods."""
    processor = MagicMock()
    processor._active_cancellation_tokens = {}

    from rasa.core.processor import MessageProcessor

    processor.register_cancellation_token = (
        MessageProcessor.register_cancellation_token.__get__(processor)
    )
    processor.unregister_cancellation_token = (
        MessageProcessor.unregister_cancellation_token.__get__(processor)
    )
    processor.cancel_background_tasks = (
        MessageProcessor.cancel_background_tasks.__get__(processor)
    )
    return processor


def test_cancel_background_tasks_signals_token():
    processor = _create_processor_with_registry()
    token = CancellationToken()
    processor.register_cancellation_token("conv-1", token)

    result = processor.cancel_background_tasks("conv-1")

    assert result is True
    assert token.is_cancelled is True


def test_cancel_background_tasks_returns_false_when_no_token():
    processor = _create_processor_with_registry()

    result = processor.cancel_background_tasks("unknown-sender")

    assert result is False


def test_unregister_cleans_up_token():
    processor = _create_processor_with_registry()
    token = CancellationToken()
    processor.register_cancellation_token("conv-1", token)
    processor.unregister_cancellation_token("conv-1")

    result = processor.cancel_background_tasks("conv-1")
    assert result is False
    assert token.is_cancelled is False


def test_register_overwrites_previous_token():
    processor = _create_processor_with_registry()
    token1 = CancellationToken()
    token2 = CancellationToken()
    processor.register_cancellation_token("conv-1", token1)
    processor.register_cancellation_token("conv-1", token2)

    result = processor.cancel_background_tasks("conv-1")

    assert result is True
    assert token2.is_cancelled is True
    assert token1.is_cancelled is False


def test_unregister_nonexistent_is_safe():
    processor = _create_processor_with_registry()
    processor.unregister_cancellation_token("nonexistent")
