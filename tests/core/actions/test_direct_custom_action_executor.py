import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import CaptureFixture, MonkeyPatch

from rasa.core.actions.action import RemoteAction, RemoteActionJSONValidator
from rasa.core.actions.direct_custom_actions_executor import DirectCustomActionExecutor
from rasa.core.agent import Agent
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    OutputChannel,
    UserMessage,
)
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

DUMMY_ACTIONS_MODULE_PATH = "data.dummy_actions_module"
DUMMY_INVALID_ACTIONS_MODULE_PATH = "data.dummy_invalid_actions_module"
DUMMY_ACTION_NAME = "my_action"
DUMMY_DOMAIN_PATH = "data/test_domains/default.yml"

ENDPOINTS_FILE_PATH = "data/test_endpoints/endpoints_actions_module.yml"


@pytest.fixture(autouse=True)
def setup(monkeypatch: MonkeyPatch):
    DirectCustomActionExecutor._actions_module_registered = False
    DirectCustomActionExecutor._create_action_executor.cache_clear()

    # Set OPENAI_API_KEY
    monkeypatch.setenv("OPENAI_API_KEY", "test-foo-bar")


@pytest.fixture
def mock_endpoint() -> EndpointConfig:
    return read_endpoint_config(ENDPOINTS_FILE_PATH, endpoint_type="action_endpoint")


@pytest.fixture
def direct_custom_action_executor(
    mock_endpoint: EndpointConfig,
) -> DirectCustomActionExecutor:
    return DirectCustomActionExecutor(
        action_name=DUMMY_ACTION_NAME, action_endpoint=mock_endpoint
    )


@pytest.fixture
def remote_action(mock_endpoint: EndpointConfig) -> RemoteAction:
    return RemoteAction(DUMMY_ACTION_NAME, mock_endpoint)


@pytest.fixture
def tracker() -> DialogueStateTracker:
    return DialogueStateTracker(sender_id="test", slots={})


@pytest.fixture
def domain() -> Domain:
    return Domain.from_file(path=DUMMY_DOMAIN_PATH)


def test_executor_initialized_with_valid_actions_module(mock_endpoint: EndpointConfig):
    try:
        DirectCustomActionExecutor(
            action_name=DUMMY_ACTION_NAME, action_endpoint=mock_endpoint
        )
    except Exception as exc:
        assert (
            False
        ), f"Instantiating 'DirectCustomActionExecutor' raised an exception {exc}"


async def test_executor_initialized_with_invalid_actions_module(
    tracker: DialogueStateTracker,
    domain: Domain,
):
    endpoint = EndpointConfig(actions_module=DUMMY_INVALID_ACTIONS_MODULE_PATH)

    # Executor should initialize successfully
    executor = DirectCustomActionExecutor(
        action_name="some_action", action_endpoint=endpoint
    )

    # Exception should be raised during run() method
    message = (
        f"You've provided the custom actions module "
        f"'{DUMMY_INVALID_ACTIONS_MODULE_PATH}' to run directly by the rasa server, "
        f"however this module does not exist. "
        f"Please check for typos in your `endpoints.yml` file."
    )
    with pytest.raises(RasaException, match=message):
        await executor.run(tracker, domain)


def test_warning_raised_for_url_and_actions_module_defined():
    endpoint = EndpointConfig(
        url="http://localhost:5055/webhook", actions_module=DUMMY_ACTIONS_MODULE_PATH
    )
    with pytest.warns(
        UserWarning, match="Both 'actions_module' and 'url' are defined."
    ):
        RemoteAction(DUMMY_ACTION_NAME, endpoint)


def test_remote_action_initializes_direct_custom_action_executor(
    remote_action: RemoteAction,
):
    assert isinstance(remote_action.executor, DirectCustomActionExecutor)


def test_remote_action_uses_action_endpoint_with_url_and_actions_module_defined():
    endpoint = EndpointConfig(
        url="http://localhost:5055/webhook", actions_module=DUMMY_ACTIONS_MODULE_PATH
    )
    remote_action = RemoteAction(DUMMY_ACTION_NAME, endpoint)
    assert isinstance(remote_action.executor, DirectCustomActionExecutor)


def test_remote_action_executor_cached(mock_endpoint: EndpointConfig):
    """
    Ensure the executor for the RemoteAction instance is being
    cached after the action endpoint is updated.

    Assertions:
    - Initially, the executor is `DirectCustomActionExecutor`.
    - After recreating the executor instance, the executor is still
      `DirectCustomActionExecutor` at the same location.
    """
    remote_action = RemoteAction(DUMMY_ACTION_NAME, mock_endpoint)
    assert isinstance(remote_action.executor, DirectCustomActionExecutor)

    initial_executor_id = id(remote_action.executor)
    remote_action.executor = remote_action._create_executor()
    assert id(remote_action.executor) == initial_executor_id


def test_direct_custom_action_executor_valid_initialization(
    direct_custom_action_executor: DirectCustomActionExecutor,
    mock_endpoint: EndpointConfig,
):
    assert direct_custom_action_executor.action_name == DUMMY_ACTION_NAME
    assert direct_custom_action_executor.action_endpoint == mock_endpoint


@pytest.mark.asyncio
async def test_executor_runs_action(
    direct_custom_action_executor: DirectCustomActionExecutor,
    tracker: DialogueStateTracker,
    domain: Domain,
):
    result = await direct_custom_action_executor.run(tracker, domain=domain)
    assert isinstance(result, dict)
    assert "events" in result


@pytest.mark.asyncio
async def test_executor_runs_action_without_response_validation(
    direct_custom_action_executor: DirectCustomActionExecutor,
    tracker: DialogueStateTracker,
    domain: Domain,
    monkeypatch: MonkeyPatch,
):
    mock_validate = MagicMock()
    monkeypatch.setattr(RemoteActionJSONValidator, "validate", mock_validate)
    await direct_custom_action_executor.run(tracker, domain=domain)
    mock_validate.assert_not_called()


async def test_executor_runs_action_invalid_actions_module(
    capsys: CaptureFixture, custom_actions_agent: Agent
):
    """
    Ensure that the inappropriately configured actions_module doesn't
    break the execution of the assistant, but logs an exception and continues.
    """
    # Set MessageProcessor to use the DirectCustomActionExecutor
    # with an invalid actions_module
    processor = custom_actions_agent.processor
    endpoint = EndpointConfig(actions_module=DUMMY_INVALID_ACTIONS_MODULE_PATH)
    processor.action_endpoint = endpoint

    # The conversation should complete successfully despite the invalid module
    message = UserMessage(text="Activate custom action.")
    response = await processor.handle_message(message)

    # Verify that the conversation completed (no exception raised)
    assert response is not None

    # Check that the error was logged
    captured = capsys.readouterr()
    error_message = "module does not exist"
    assert error_message in captured.out or error_message in captured.err


def test_action_executor_is_being_cached(mock_endpoint: EndpointConfig):
    executor_1 = DirectCustomActionExecutor(
        action_name=DUMMY_ACTION_NAME, action_endpoint=mock_endpoint
    )
    executor_2 = DirectCustomActionExecutor(
        action_name=DUMMY_ACTION_NAME, action_endpoint=mock_endpoint
    )
    assert executor_1.action_executor == executor_2.action_executor


@pytest.mark.asyncio
async def test_conversation_completes_with_invalid_module():
    """Test that conversation completes properly when action module doesn't exist."""
    # Create an action with an invalid action module
    endpoint = EndpointConfig(actions_module="nonexistent_module")
    action = RemoteAction("test_action", endpoint)

    # Create a simple domain and tracker
    domain = Domain.empty()
    tracker = DialogueStateTracker("test_sender", [])

    # Test that the action can be created without exception
    assert isinstance(action, RemoteAction)
    assert isinstance(action.executor, DirectCustomActionExecutor)

    # Test that running the action raises the expected exception
    # This simulates what happens in the processor's _run_action method
    with pytest.raises(RasaException, match="module does not exist"):
        await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )


async def test_custom_actions_hot_reloading():
    def create_action_code(value: str) -> str:
        return f"""from typing import Any, Dict
from rasa_sdk.interfaces import Action
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher

class CustomAction(Action):
    def name(self) -> str:
        return "custom_action"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> Any:
        return [{{"event": "slot", "name": "test_slot", "value": "{value}"}}]
"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir_path = Path(tmpdirname)

        # Create a subdirectory for the module
        module_name = "custom_actions_test_module"
        action_module_path = tmpdir_path / module_name
        action_module_path.mkdir(parents=True, exist_ok=True)

        # Create __init__.py to make it a package
        action_module_init = action_module_path / "__init__.py"
        action_module_init.touch()

        # Add the temporary directory to sys.path
        sys.path.insert(0, str(tmpdir_path))

        # Create the action file inside the module
        action_file = action_module_path / "custom_action.py"
        initial_value = "initial_value"
        action_file.write_text(create_action_code(initial_value))

        # Create an endpoint and executor to run the initial custom action
        endpoint = EndpointConfig(actions_module=module_name)
        executor = DirectCustomActionExecutor("custom_action", endpoint)
        tracker = DialogueStateTracker("default", [])
        domain = Domain.empty()
        result_initial = await executor.run(tracker, domain)
        assert result_initial["events"][0]["value"] == initial_value

        # Modify the custom action file with a new value
        modified_value = "modified_value"
        action_file.write_text(create_action_code(modified_value))

        # Manually update the file's modification time to ensure it's detectable
        new_time = time.time() + 60  # Set time to 60 seconds in the future
        os.utime(action_file, (new_time, new_time))

        # Run the custom action with the modified value
        executor = DirectCustomActionExecutor("custom_action", endpoint)
        result_modified = await executor.run(tracker, domain)
        assert result_modified["events"][0]["value"] == modified_value


# ---------------------------------------------------------------------------
# run_streaming() — stream_chunk routing tests
# ---------------------------------------------------------------------------


class _CapturingOutputChannel(OutputChannel):
    """Records both incremental text chunks and full send_response calls."""

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self.responses: list[dict[str, Any]] = []
        self.started: bool = False
        self.ended: bool = False

    @property
    def supports_streaming(self) -> bool:
        return True

    async def send_text_message(self, recipient_id: str, text: str, **kw: Any) -> None:
        pass

    async def send_response_chunk_start(self, recipient_id: str, **kw: Any) -> None:
        self.started = True

    async def send_response_chunk(
        self, recipient_id: str, chunk: str, **kw: Any
    ) -> None:
        self.chunks.append(chunk)

    async def send_response_chunk_end(self, recipient_id: str, **kw: Any) -> None:
        self.ended = True

    async def send_response(self, recipient_id: str, message: dict[str, Any]) -> None:
        self.responses.append(message)


def _make_executor_with_error(
    exc: Exception,
) -> tuple[DirectCustomActionExecutor, Any]:
    """Return an executor whose ActionExecutor.run_streaming() raises *exc*.

    The ``_on_task_done`` callback in ``DirectCustomActionExecutor.run_streaming``
    catches the task failure and puts ``{"event": "_error", "exc": exc}`` in the
    sink, which the consumer loop then re-raises.
    """
    from rasa_sdk.executor import ActionExecutor

    endpoint = MagicMock(spec=EndpointConfig)
    endpoint.actions_module = "actions"

    executor = DirectCustomActionExecutor.__new__(DirectCustomActionExecutor)
    executor.action_name = "action_test"
    executor.action_endpoint = endpoint
    executor.action_executor = MagicMock(spec=ActionExecutor)
    executor.action_executor.reload = MagicMock()

    async def fake_run_streaming_raises(
        action_call: dict[str, Any], sink: asyncio.Queue
    ) -> None:
        raise exc

    executor.action_executor.run_streaming = fake_run_streaming_raises

    tracker = MagicMock()
    tracker.sender_id = "test_user"
    tracker.current_state = MagicMock(return_value={})

    return executor, tracker


def _make_executor_with_events(
    events: list[dict[str, Any]],
    final_result: dict[str, Any],
) -> tuple[DirectCustomActionExecutor, Any]:
    """Return a (DirectCustomActionExecutor, tracker_mock) pair whose internal
    ActionExecutor.run_streaming() emits *events* then a ``stream_done`` sentinel,
    mirroring the real SDK protocol."""
    from rasa_sdk.executor import ActionExecutor

    endpoint = MagicMock(spec=EndpointConfig)
    endpoint.actions_module = "actions"

    executor = DirectCustomActionExecutor.__new__(DirectCustomActionExecutor)
    executor.action_name = "action_test"
    executor.action_endpoint = endpoint
    executor.action_executor = MagicMock(spec=ActionExecutor)
    executor.action_executor.reload = MagicMock()

    class _FakeResult:
        def model_dump(self) -> dict[str, Any]:
            return final_result

    async def fake_run_streaming(
        action_call: dict[str, Any], sink: asyncio.Queue
    ) -> _FakeResult:
        for event in events:
            await sink.put(event)
        result = _FakeResult()
        # The consumer loop in DirectCustomActionExecutor.run_streaming() only
        # exits when it receives a "stream_done" event carrying the result object.
        # Without this sentinel the loop blocks forever, closing the event loop.
        await sink.put({"event": "stream_done", "result": result})
        return result

    executor.action_executor.run_streaming = fake_run_streaming

    tracker = MagicMock()
    tracker.sender_id = "test_user"
    tracker.current_state = MagicMock(return_value={})

    return executor, tracker


@pytest.mark.asyncio
async def test_run_streaming_text_chunk_calls_send_response_chunk(
    domain: Domain,
) -> None:
    """stream_chunk events with only text are forwarded via send_response_chunk."""
    events = [
        {"event": "stream_start"},
        {"event": "stream_chunk", "text": "Hello"},
        {"event": "stream_chunk", "text": " world"},
        {"event": "stream_end"},
    ]
    final_result: dict[str, Any] = {"events": [], "responses": []}
    executor, tracker = _make_executor_with_events(events, final_result)

    channel = _CapturingOutputChannel()
    with patch.object(executor, "register_actions_from_a_module"):
        result = await executor.run_streaming(
            tracker=tracker,
            domain=domain,
            output_channel=channel,
        )

    assert channel.started is True
    assert channel.chunks == ["Hello", " world"]
    assert channel.ended is True
    assert channel.responses == []
    assert result == final_result


@pytest.mark.asyncio
async def test_run_streaming_rich_chunk_calls_send_response(
    domain: Domain,
) -> None:
    """stream_chunk events with rich content are delivered via send_response."""
    buttons = [{"title": "Yes", "payload": "/affirm"}]
    events = [
        {"event": "stream_start"},
        {"event": "stream_chunk", "text": "Pick one:", "buttons": buttons},
        {"event": "stream_end"},
    ]
    final_result: dict[str, Any] = {"events": [], "responses": []}
    executor, tracker = _make_executor_with_events(events, final_result)

    channel = _CapturingOutputChannel()
    with patch.object(executor, "register_actions_from_a_module"):
        result = await executor.run_streaming(
            tracker=tracker,
            domain=domain,
            output_channel=channel,
        )

    assert channel.chunks == [], "send_response_chunk should not have been called"
    assert len(channel.responses) == 1
    assert channel.responses[0]["text"] == "Pick one:"
    assert channel.responses[0]["buttons"] == buttons
    assert result == final_result


@pytest.mark.asyncio
async def test_remote_action_passes_final_responses_to_utter_on_streaming_path(
    domain: Domain,
) -> None:
    """RemoteAction.run() still calls _utter_responses with final_result.responses
    on the streaming path.

    An action may legitimately stream tokens via stream_chunk() AND dispatch a
    separate, non-streamed message (e.g. a follow-up prompt or a rich card) via
    utter_message().  Suppressing final_result.responses on the streaming path
    would silently drop those legitimate additional messages and also prevent
    BotUttered tracker events from being created.

    Duplicates arising from an action calling BOTH stream_chunk() and
    utter_message() for the same content are an action-code bug; they are
    flagged at development time by warn_on_duplicate_streamed_responses().
    """
    full_response = {"text": "Is there anything else I can help with?"}
    events = [
        {"event": "stream_start"},
        {"event": "stream_chunk", "text": "Here is your answer."},
        {"event": "stream_end"},
    ]
    # The action streams the main reply and also dispatches a separate follow-up.
    final_result: dict[str, Any] = {"events": [], "responses": [full_response]}
    executor, tracker = _make_executor_with_events(events, final_result)

    channel = _CapturingOutputChannel()
    nlg = MagicMock()

    remote_action = RemoteAction.__new__(RemoteAction)
    remote_action._name = "action_test"
    remote_action.action_endpoint = MagicMock()
    remote_action.executor = executor

    with (
        patch.object(executor, "register_actions_from_a_module"),
        patch.object(
            remote_action, "_utter_responses", wraps=remote_action._utter_responses
        ) as mock_utter,
    ):
        await remote_action.run(
            output_channel=channel,
            nlg=nlg,
            tracker=tracker,
            domain=domain,
        )

    # _utter_responses must receive the actual responses so that:
    # 1. Legitimate follow-up messages are delivered to the channel.
    # 2. BotUttered tracker events are created for every response.
    mock_utter.assert_called_once()
    uttered_responses = mock_utter.call_args[0][0]
    assert uttered_responses == [full_response]
    # The streamed tokens arrive via stream_chunk.
    assert channel.chunks == ["Here is your answer."]


@pytest.mark.asyncio
async def test_run_streaming_warns_on_duplicate_response(domain: Domain) -> None:
    """warn_on_duplicate_streamed_responses is called when final_result.responses
    contains a payload that was already delivered as a stream_chunk event."""
    duplicate: dict[str, Any] = {"text": "Hello world"}
    events = [
        {"event": "stream_start"},
        {"event": "stream_chunk", "text": "Hello world"},
        {"event": "stream_end"},
    ]
    final_result: dict[str, Any] = {"events": [], "responses": [duplicate]}
    executor, tracker = _make_executor_with_events(events, final_result)

    channel = _CapturingOutputChannel()
    with (
        patch.object(executor, "register_actions_from_a_module"),
        patch(
            "rasa.core.actions.direct_custom_actions_executor.warn_on_duplicate_streamed_responses"
        ) as mock_warn,
    ):
        await executor.run_streaming(
            tracker=tracker,
            domain=domain,
            output_channel=channel,
        )

    mock_warn.assert_called_once_with(
        action_name="action_test",
        streamed_payloads=[duplicate],
        final_responses=[duplicate],
    )


@pytest.mark.asyncio
async def test_run_streaming_re_raises_exception_from_action_task(
    domain: Domain,
) -> None:
    """run_streaming() propagates any exception raised by the SDK action task.

    When ``ActionExecutor.run_streaming()`` raises (e.g. an
    ``ActionExecutionRejection`` or any unhandled error), the
    ``_on_task_done`` callback puts a ``{"event": "_error", "exc": ...}``
    sentinel in the sink, and the consumer loop re-raises it.  This ensures
    callers receive the original exception rather than a silent hang or a
    generic timeout.
    """
    original_exc = RasaException("Action execution failed unexpectedly.")
    executor, tracker = _make_executor_with_error(original_exc)

    channel = _CapturingOutputChannel()
    with patch.object(executor, "register_actions_from_a_module"):
        with pytest.raises(
            RasaException, match="Action execution failed unexpectedly."
        ):
            await executor.run_streaming(
                tracker=tracker,
                domain=domain,
                output_channel=channel,
            )


@pytest.mark.asyncio
async def test_run_streaming_always_includes_domain(
    domain: Domain,
) -> None:
    """The direct executor always passes the domain regardless of include_domain.

    Unlike the HTTP/gRPC executors, ``DirectCustomActionExecutor`` runs the SDK
    in-process.  There is no network payload overhead, so the domain is always
    serialised into the action call — even when ``include_domain=False`` (the
    default).  This prevents the SDK from raising
    ``ActionMissingDomainException`` on every first call.
    """
    captured_calls: list[dict] = []

    async def fake_run_streaming(
        action_call: dict[str, Any], sink: asyncio.Queue
    ) -> None:
        captured_calls.append(action_call)
        await sink.put({"event": "stream_done", "result": None})

    endpoint = MagicMock(spec=EndpointConfig)
    endpoint.actions_module = "actions"
    executor = DirectCustomActionExecutor.__new__(DirectCustomActionExecutor)
    executor.action_name = "action_test"
    executor.action_endpoint = endpoint
    from rasa_sdk.executor import ActionExecutor

    executor.action_executor = MagicMock(spec=ActionExecutor)
    executor.action_executor.reload = MagicMock()
    executor.action_executor.run_streaming = fake_run_streaming

    tracker = MagicMock()
    tracker.sender_id = "test_user"
    tracker.current_state = MagicMock(return_value={})

    channel = _CapturingOutputChannel()
    with patch.object(executor, "register_actions_from_a_module"):
        await executor.run_streaming(
            tracker=tracker,
            domain=domain,
            output_channel=channel,
            include_domain=False,
        )

    assert len(captured_calls) == 1
    assert "domain" in captured_calls[0], (
        "Domain must always be included in direct executor streaming calls, "
        "regardless of include_domain flag"
    )


@pytest.mark.asyncio
async def test_run_streaming_translates_action_missing_domain_to_domain_not_found(
    domain: Domain,
) -> None:
    """ActionMissingDomainException from the SDK is converted to DomainNotFound.

    Even though the direct executor now always includes the domain, if the SDK
    still raises ``ActionMissingDomainException`` for any reason,
    ``_consume_stream_events`` must translate it into ``DomainNotFound`` so that
    ``RetryCustomActionExecutor.run_streaming`` can handle it gracefully.
    """
    from rasa_sdk.interfaces import ActionMissingDomainException

    from rasa.core.actions.action_exceptions import DomainNotFound

    original_exc = ActionMissingDomainException(
        "Missing domain context, assistant will retry."
    )
    executor, tracker = _make_executor_with_error(original_exc)

    channel = _CapturingOutputChannel()
    with patch.object(executor, "register_actions_from_a_module"):
        with pytest.raises(DomainNotFound):
            await executor.run_streaming(
                tracker=tracker,
                domain=domain,
                output_channel=channel,
            )


@pytest.mark.asyncio
async def test_run_streaming_re_raises_exception_via_async_fallback_when_queue_full(
    domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    """_on_task_done uses an async fallback when put_nowait raises QueueFull.

    When the sink queue is full at the moment the executor task fails,
    ``put_nowait`` raises ``asyncio.QueueFull``.  The ``_on_task_done``
    callback must then create a background ``asyncio.create_task`` that
    calls ``_put_error_on_sink``.  The exception must still propagate to
    the caller via the consumer loop.
    """

    class _FirstPutNowaitRaisesFullQueue(asyncio.Queue):
        """Queue whose first ``put_nowait`` call raises ``QueueFull``.

        This forces ``_on_task_done`` to take the async fallback path
        (``asyncio.create_task(_put_error_on_sink(exc))``).  Subsequent
        ``put_nowait`` / ``await put`` calls work normally so the fallback
        task can actually deliver the error sentinel to the consumer loop.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._put_nowait_calls = 0

        def put_nowait(self, item: Any) -> None:
            self._put_nowait_calls += 1
            if self._put_nowait_calls == 1:
                raise asyncio.QueueFull()
            super().put_nowait(item)

    monkeypatch.setattr(asyncio, "Queue", _FirstPutNowaitRaisesFullQueue)

    original_exc = RasaException("Action failed while sink was full.")
    executor, tracker = _make_executor_with_error(original_exc)

    channel = _CapturingOutputChannel()
    with patch.object(executor, "register_actions_from_a_module"):
        with pytest.raises(RasaException, match="Action failed while sink was full."):
            await executor.run_streaming(
                tracker=tracker,
                domain=domain,
                output_channel=channel,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception, exception_type, match_msg",
    [
        (RuntimeError("channel write failed"), RuntimeError, "channel write failed"),
        (
            asyncio.CancelledError("client disconnected"),
            asyncio.CancelledError,
            "client disconnected",
        ),
    ],
)
async def test_run_streaming_closes_chunk_session_on_channel_error_mid_stream(
    domain: Domain,
    exception: Any,
    exception_type: Any,
    match_msg: str,
) -> None:
    """send_response_chunk raising mid-stream closes the session and propagates.

    When the output channel raises while forwarding a text chunk, the executor
    must (1) re-raise the original exception so the caller can record the
    failure, and (2) still close the chunk session so the client is not left
    waiting indefinitely for a chunk_end that never arrives.
    """
    events = [
        {"event": "stream_start"},
        {"event": "stream_chunk", "text": "Hello"},
        {"event": "stream_chunk", "text": " world"},
        {"event": "stream_end"},
    ]
    executor, tracker = _make_executor_with_events(events, {})

    channel = _CapturingOutputChannel()
    channel.send_response_chunk = AsyncMock(side_effect=exception)

    with patch.object(executor, "register_actions_from_a_module"):
        with pytest.raises(exception_type, match=match_msg):
            await executor.run_streaming(
                tracker=tracker,
                domain=domain,
                output_channel=channel,
            )

    assert (
        channel.ended is True
    ), "chunk session must be closed after a mid-stream channel error"


@pytest.mark.asyncio
async def test_run_streaming_closes_chunk_session_when_chunk_start_raises(
    domain: Domain,
) -> None:
    """send_response_chunk_start raising still triggers chunk_end cleanup.

    The None sentinel marks the session as 'attempted' before the call so
    that cleanup fires even when chunk_start itself raises — the client may
    have received a partial stream_start frame and still needs a close signal.
    """
    events = [
        {"event": "stream_start"},
        {"event": "stream_chunk", "text": "Hello"},
        {"event": "stream_end"},
    ]
    executor, tracker = _make_executor_with_events(events, {})

    channel = _CapturingOutputChannel()
    channel.send_response_chunk_start = AsyncMock(
        side_effect=RuntimeError("connection dropped")
    )

    with patch.object(executor, "register_actions_from_a_module"):
        with pytest.raises(RuntimeError, match="connection dropped"):
            await executor.run_streaming(
                tracker=tracker,
                domain=domain,
                output_channel=channel,
            )

    assert not channel.started, "started must remain False since chunk_start raised"
    assert (
        channel.ended is True
    ), "chunk_end must be sent as cleanup even when chunk_start itself raised"


@pytest.mark.asyncio
async def test_run_streaming_cancels_executor_task_on_channel_error(
    domain: Domain,
) -> None:
    """The background executor task is cancelled when the consumer exits early.

    If the output channel raises mid-stream, _consume_stream_events exits
    before the SDK action task has finished.  Without cancellation the task
    would block forever waiting to put events onto the full sink queue.
    """
    from rasa_sdk.executor import ActionExecutor

    endpoint = MagicMock(spec=EndpointConfig)
    endpoint.actions_module = "actions"

    executor = DirectCustomActionExecutor.__new__(DirectCustomActionExecutor)
    executor.action_name = "action_test"
    executor.action_endpoint = endpoint
    executor.action_executor = MagicMock(spec=ActionExecutor)
    executor.action_executor.reload = MagicMock()

    task_was_cancelled = asyncio.Event()

    async def eternal_run_streaming(
        action_call: dict[str, Any], sink: asyncio.Queue
    ) -> None:
        try:
            await sink.put({"event": "stream_start"})
            await asyncio.sleep(60)  # blocks until cancelled
        except asyncio.CancelledError:
            task_was_cancelled.set()
            raise

    executor.action_executor.run_streaming = eternal_run_streaming

    tracker = MagicMock()
    tracker.sender_id = "test_user"
    tracker.current_state = MagicMock(return_value={})

    channel = _CapturingOutputChannel()
    channel.send_response_chunk_start = AsyncMock(
        side_effect=RuntimeError("channel broken")
    )

    with patch.object(executor, "register_actions_from_a_module"):
        with pytest.raises(RuntimeError, match="channel broken"):
            await executor.run_streaming(
                tracker=tracker,
                domain=domain,
                output_channel=channel,
            )

    assert (
        task_was_cancelled.is_set()
    ), "executor task must be cancelled so it does not block on a full sink queue"


@pytest.mark.asyncio
async def test_run_streaming_cancels_executor_task_on_external_cancellation(
    domain: Domain,
) -> None:
    """The background executor task is cancelled when the outer task is cancelled.

    Uses real asyncio.Task.cancel() to simulate a client disconnect so that
    CancelledError is injected by the event loop rather than via side_effect,
    verifying the full asyncio cancellation path end-to-end.
    """
    from rasa_sdk.executor import ActionExecutor

    endpoint = MagicMock(spec=EndpointConfig)
    endpoint.actions_module = "actions"

    executor = DirectCustomActionExecutor.__new__(DirectCustomActionExecutor)
    executor.action_name = "action_test"
    executor.action_endpoint = endpoint
    executor.action_executor = MagicMock(spec=ActionExecutor)
    executor.action_executor.reload = MagicMock()

    task_was_cancelled = asyncio.Event()
    consumer_reached_chunk = asyncio.Event()

    async def controlled_run_streaming(
        action_call: dict[str, Any], sink: asyncio.Queue
    ) -> None:
        try:
            await sink.put({"event": "stream_start"})
            await sink.put({"event": "stream_chunk", "text": "Hello"})
            await asyncio.sleep(60)  # blocks until cancelled
        except asyncio.CancelledError:
            task_was_cancelled.set()
            raise

    executor.action_executor.run_streaming = controlled_run_streaming

    tracker = MagicMock()
    tracker.sender_id = "test_user"
    tracker.current_state = MagicMock(return_value={})

    channel = _CapturingOutputChannel()

    # Pause inside send_response_chunk long enough to cancel the outer task.
    async def slow_chunk(recipient_id: str, chunk: str, **kw: Any) -> None:
        consumer_reached_chunk.set()
        await asyncio.sleep(60)  # will be interrupted by cancellation

    channel.send_response_chunk = slow_chunk

    async def run() -> None:
        with patch.object(executor, "register_actions_from_a_module"):
            await executor.run_streaming(
                tracker=tracker,
                domain=domain,
                output_channel=channel,
            )

    outer_task = asyncio.create_task(run())
    # Wait until the consumer is mid-chunk, then cancel.
    await consumer_reached_chunk.wait()
    outer_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await outer_task

    assert (
        channel.ended is True
    ), "chunk session must be closed after external task cancellation"
    assert (
        task_was_cancelled.is_set()
    ), "background executor task must be cancelled when the outer task is cancelled"
