import asyncio
from unittest.mock import MagicMock

import pytest

from rasa.core.actions.action import ActionBotResponse
from rasa.core.channels import CollectingOutputChannel
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.core.nlg import CallbackNaturalLanguageGenerator
from rasa.core.nlg.callback import nlg_request_format
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.fixture
def mock_nlg_endpoint() -> MagicMock:
    _mock_nlg_endpoint = MagicMock()

    future = asyncio.Future()
    future.set_result({})

    _mock_nlg_endpoint.request = MagicMock()
    _mock_nlg_endpoint.request.return_value = future
    return _mock_nlg_endpoint


async def test_action_bot_response_callback_nlg(
    domain_with_response_ids: Domain,
    default_tracker: DialogueStateTracker,
    mock_nlg_endpoint: MagicMock,
):
    """Test the response returned by the callback NLG endpoint."""
    callback_nlg = CallbackNaturalLanguageGenerator(mock_nlg_endpoint)

    output_channel = CollectingOutputChannel()

    events = await ActionBotResponse("utter_one_id").run(
        output_channel, callback_nlg, default_tracker, domain_with_response_ids
    )

    expected_body = nlg_request_format(
        "utter_one_id",
        default_tracker,
        output_channel.name(),
        response_id="1",
    )

    mock_nlg_endpoint.request.assert_called_once_with(
        method="post", json=expected_body, timeout=DEFAULT_REQUEST_TIMEOUT
    )

    assert len(events) == 1
    assert events[0].metadata == {"utter_action": "utter_one_id"}


async def test_action_bot_response_callback_with_multiple_response_id(
    domain_with_response_ids: Domain,
    default_tracker: DialogueStateTracker,
    mock_nlg_endpoint: MagicMock,
) -> None:
    callback_nlg = CallbackNaturalLanguageGenerator(mock_nlg_endpoint)

    output_channel = CollectingOutputChannel()

    events = await ActionBotResponse("utter_multiple_ids").run(
        output_channel, callback_nlg, default_tracker, domain_with_response_ids
    )

    expected_body = nlg_request_format(
        "utter_multiple_ids",
        default_tracker,
        output_channel.name(),
        response_id="2",
    )

    mock_nlg_endpoint.request.assert_called_once_with(
        method="post", json=expected_body, timeout=DEFAULT_REQUEST_TIMEOUT
    )

    assert len(events) == 1
    assert events[0].metadata == {"utter_action": "utter_multiple_ids"}


async def test_action_bot_response_with_empty_response_id_set(
    domain_with_response_ids: Domain,
    default_tracker: DialogueStateTracker,
    mock_nlg_endpoint: MagicMock,
) -> None:
    callback_nlg = CallbackNaturalLanguageGenerator(mock_nlg_endpoint)

    output_channel = CollectingOutputChannel()

    events = await ActionBotResponse("utter_no_id").run(
        output_channel, callback_nlg, default_tracker, domain_with_response_ids
    )

    expected_body = nlg_request_format(
        "utter_no_id",
        default_tracker,
        output_channel.name(),
        response_id=None,
    )

    mock_nlg_endpoint.request.assert_called_once_with(
        method="post", json=expected_body, timeout=DEFAULT_REQUEST_TIMEOUT
    )

    assert len(events) == 1
    assert events[0].metadata == {"utter_action": "utter_no_id"}


async def test_action_bot_response_with_non_existing_id_mapping(
    domain_with_response_ids: Domain,
    default_tracker: DialogueStateTracker,
    mock_nlg_endpoint: MagicMock,
) -> None:
    callback_nlg = CallbackNaturalLanguageGenerator(mock_nlg_endpoint)

    output_channel = CollectingOutputChannel()

    events = await ActionBotResponse("utter_non_existing").run(
        output_channel, callback_nlg, default_tracker, domain_with_response_ids
    )

    expected_body = nlg_request_format(
        "utter_non_existing",
        default_tracker,
        output_channel.name(),
        response_id=None,
    )

    mock_nlg_endpoint.request.assert_called_once_with(
        method="post", json=expected_body, timeout=DEFAULT_REQUEST_TIMEOUT
    )

    assert len(events) == 1
    assert events[0].metadata == {"utter_action": "utter_non_existing"}
