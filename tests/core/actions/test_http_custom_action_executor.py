import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from rasa.core.actions.action_exceptions import ActionExecutionRejection, DomainNotFound
from rasa.core.actions.custom_action_executor import (
    ActionResult,
    ActionResultType,
    RetryCustomActionExecutor,
)
from rasa.core.actions.http_custom_action_executor import HTTPCustomActionExecutor
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import ClientResponseError, EndpointConfig


@pytest.fixture
def mock_endpoint() -> EndpointConfig:
    endpoint = MagicMock(spec=EndpointConfig)
    endpoint.url = "http://localhost:5055/webhook"
    endpoint.kwargs = {}
    return endpoint


@pytest.fixture
def http_executor(mock_endpoint: EndpointConfig) -> HTTPCustomActionExecutor:
    return HTTPCustomActionExecutor("test_action", mock_endpoint)


@pytest.fixture
def tracker() -> DialogueStateTracker:
    return DialogueStateTracker.from_events("test", evts=[SlotSet("foo", "bar")])


@pytest.fixture
def domain() -> Domain:
    return Domain.from_dict({"responses": {}})


@pytest.mark.asyncio
async def test_run_with_result_returns_retry_on_449_status(
    http_executor: HTTPCustomActionExecutor,
    tracker: DialogueStateTracker,
    domain: Domain,
    mock_endpoint: EndpointConfig,
) -> None:
    mock_endpoint.request = AsyncMock(return_value={"missing_domain": True})

    result = await http_executor.run_with_result(tracker, domain, include_domain=False)

    assert isinstance(result, ActionResult)
    assert result.result_type == ActionResultType.RETRY_WITH_DOMAIN
    assert result.response is None


@pytest.mark.asyncio
async def test_run_with_result_returns_success_on_200_status(
    http_executor: HTTPCustomActionExecutor,
    tracker: DialogueStateTracker,
    domain: Domain,
    mock_endpoint: EndpointConfig,
) -> None:
    response_data = {"events": [], "responses": []}

    async def mock_request(*args, **kwargs):
        return response_data

    mock_endpoint.request = AsyncMock(side_effect=mock_request)

    result = await http_executor.run_with_result(tracker, domain, include_domain=False)

    assert isinstance(result, ActionResult)
    assert result.result_type == ActionResultType.SUCCESS
    assert result.response == response_data


@pytest.mark.asyncio
async def test_run_returns_empty_dict_for_missing_domain(
    http_executor: HTTPCustomActionExecutor,
    tracker: DialogueStateTracker,
    domain: Domain,
    mock_endpoint: EndpointConfig,
) -> None:
    mock_endpoint.request = AsyncMock(return_value={"missing_domain": True})

    result = await http_executor.run(tracker, domain, include_domain=False)
    assert result == {}


@pytest.mark.asyncio
async def test_run_with_result_handles_other_errors(
    http_executor: HTTPCustomActionExecutor,
    tracker: DialogueStateTracker,
    domain: Domain,
    mock_endpoint: EndpointConfig,
) -> None:
    # Test 400 status (ActionExecutionRejection)
    async def mock_400_request(*args, **kwargs):
        body_text = json.dumps({"action_name": "test_action", "error": "Invalid input"})
        error = ClientResponseError(400, "BAD_REQUEST", body_text)
        # ClientResponseError expects the text attribute
        error.text = body_text
        raise error

    mock_endpoint.request = AsyncMock(side_effect=mock_400_request)

    with pytest.raises(ActionExecutionRejection) as exc_info:
        await http_executor.run_with_result(tracker, domain, include_domain=False)
    assert exc_info.value.action_name == "test_action"
    assert "Invalid input" in str(exc_info.value)

    # Test 404 status
    async def mock_404_request(*args, **kwargs):
        raise ClientResponseError(404, "NOT_FOUND", "Action not found")

    mock_endpoint.request = AsyncMock(side_effect=mock_404_request)

    with pytest.raises(RasaException) as exc_info:
        await http_executor.run_with_result(tracker, domain, include_domain=False)
    assert "not found" in str(exc_info.value)

    # Test connection error
    async def mock_connection_error(*args, **kwargs):
        raise aiohttp.ClientConnectionError("Connection failed")

    mock_endpoint.request = AsyncMock(side_effect=mock_connection_error)

    with pytest.raises(RasaException) as exc_info:
        await http_executor.run_with_result(tracker, domain, include_domain=False)
    assert "Couldn't connect" in str(exc_info.value)


@pytest.mark.asyncio
async def test_retry_executor_handles_successful_first_call(
    mock_endpoint: EndpointConfig,
    tracker: DialogueStateTracker,
    domain: Domain,
) -> None:
    response_data = {"events": [], "responses": []}
    mock_endpoint.request = AsyncMock(return_value=response_data)

    http_executor = HTTPCustomActionExecutor("test_action", mock_endpoint)
    retry_executor = RetryCustomActionExecutor(http_executor)

    result = await retry_executor.run(tracker, domain, include_domain=False)

    assert result == response_data
    assert mock_endpoint.request.call_count == 1


@pytest.mark.asyncio
async def test_retry_executor_retries_on_missing_domain(
    mock_endpoint: EndpointConfig,
    tracker: DialogueStateTracker,
    domain: Domain,
) -> None:
    response_data = {"events": [], "responses": []}

    mock_endpoint.request = AsyncMock(
        side_effect=[
            {"missing_domain": True},
            response_data,
        ]
    )

    http_executor = HTTPCustomActionExecutor("test_action", mock_endpoint)
    retry_executor = RetryCustomActionExecutor(http_executor)

    result = await retry_executor.run(tracker, domain, include_domain=False)

    assert result == response_data
    assert mock_endpoint.request.call_count == 2

    # Verify first call was without domain, second with domain
    calls = mock_endpoint.request.call_args_list
    assert "domain" not in calls[0][1]["json"]  # First call should exclude domain
    assert "domain" in calls[1][1]["json"]  # Second call should include domain


@pytest.mark.asyncio
async def test_retry_executor_raises_after_two_missing_domain_responses(
    mock_endpoint: EndpointConfig,
    tracker: DialogueStateTracker,
    domain: Domain,
) -> None:
    mock_endpoint.request = AsyncMock(return_value={"missing_domain": True})

    http_executor = HTTPCustomActionExecutor("test_action", mock_endpoint)
    retry_executor = RetryCustomActionExecutor(http_executor)

    # Execute the action - should raise after second attempt
    with pytest.raises(DomainNotFound):
        await retry_executor.run(tracker, domain, include_domain=False)

    # Verify both attempts were made
    assert mock_endpoint.request.call_count == 2
