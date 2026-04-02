import asyncio
import base64
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.core.channels.inspector import (
    InspectorInputChannel,
    InspectorTextOutputChannel,
    InspectorTrackerUpdatePlugin,
    InspectorVoiceOutputChannel,
    SocketIOVoiceWebsocketAdapter,
    does_need_action_prediction,
    tracker_as_dump,
)
from rasa.core.channels.voice_stream.voice_channel import (
    DEFAULT_INTERRUPTION_MIN_WORDS,
    ContinueConversationAction,
    EndConversationAction,
    NewAudioAction,
)
from rasa.shared.core.constants import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME
from rasa.shared.core.events import ActionExecuted, SessionStarted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.fixture
def inspector_input() -> InspectorInputChannel:
    sio = AsyncMock()
    channel = InspectorInputChannel("", {}, {})
    channel.sio_server = sio
    return channel


@pytest.fixture
def _set_call_state():
    """Set up the call_state context var for tests."""
    from rasa.core.channels.voice_stream.call_state import CallState, _call_state

    state = CallState(
        internal_queue=asyncio.Queue(),
        asr_event_queue=asyncio.Queue(),
    )
    token = _call_state.set(state)
    yield state
    _call_state.reset(token)


def test_tracker_as_dump_runs(default_tracker: DialogueStateTracker) -> None:
    assert tracker_as_dump(default_tracker) is not None


def test_tracker_as_dump_only_returns_last_session_on_tracker(
    default_tracker: DialogueStateTracker,
) -> None:
    default_tracker.update(SessionStarted())
    default_tracker.update(ActionExecuted(ACTION_SESSION_START_NAME))
    default_tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    default_tracker.update(SessionStarted())
    default_tracker.update(ActionExecuted(ACTION_SESSION_START_NAME))
    default_tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    default_tracker.update(SessionStarted())
    default_tracker.update(ActionExecuted(ACTION_SESSION_START_NAME))
    default_tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    loaded = tracker_as_dump(default_tracker)

    # there should be one action listen and one action session start
    assert len(loaded.get("events")) == 2


async def test_tracker_update_plugin_triggers_after_new_user_message(
    inspector_input: InspectorInputChannel,
    default_tracker: DialogueStateTracker,
) -> None:
    default_tracker.update(UserUttered("hello world"))

    plugin = InspectorTrackerUpdatePlugin(inspector_input)
    plugin.after_new_user_message(default_tracker)

    assert len(plugin.tasks) == 1
    assert not plugin.tasks[0].done()
    await asyncio.sleep(0.1)
    assert plugin.tasks[0].done()

    dump = tracker_as_dump(default_tracker)

    expected_calls = [
        call(
            "tracker",
            dump,
            room=default_tracker.sender_id,
        ),
    ]
    inspector_input.sio_server.emit.assert_has_calls(expected_calls, any_order=False)


async def test_tracker_update_plugin_triggers_after_action_executed(
    inspector_input: InspectorInputChannel,
    default_tracker: DialogueStateTracker,
) -> None:
    default_tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    plugin = InspectorTrackerUpdatePlugin(inspector_input)
    plugin.after_action_executed(default_tracker)

    assert len(plugin.tasks) == 1
    assert not plugin.tasks[0].done()
    await asyncio.sleep(0.1)
    assert plugin.tasks[0].done()

    dump = tracker_as_dump(default_tracker)

    expected_calls = [
        call(
            "tracker",
            dump,
            room=default_tracker.sender_id,
        ),
    ]
    inspector_input.sio_server.emit.assert_has_calls(expected_calls, any_order=False)


async def test_inspector_handle_tracker_update(
    inspector_input: InspectorInputChannel,
    default_tracker: DialogueStateTracker,
    agent_with_flows: Agent,
) -> None:
    # Use a unique sender_id to avoid cross-test pollution: agent_with_flows is
    # session-scoped, so the tracker store is shared; a fixed id can collide with
    # other tests or leave state that makes this test flaky.
    sender_id = f"test_inspector_handle_tracker_update_{uuid.uuid4().hex}"
    default_tracker.sender_id = sender_id
    default_tracker.update(UserUttered("foo bar"))
    await agent_with_flows.tracker_store.save(default_tracker)

    inspector_input.agent = agent_with_flows

    data = {
        "sender_id": sender_id,
        "events": [
            UserUttered("hello world").as_dict(),
            ActionExecuted(ACTION_LISTEN_NAME).as_dict(),
        ],
    }
    await inspector_input.handle_tracker_update("some_sid", data)
    # Allow async tasks from hooks to complete (small grace period)
    await asyncio.sleep(0.05)

    assert len(inspector_input.sio_server.emit.call_args_list) == 1
    call = inspector_input.sio_server.emit.call_args_list[0]
    assert call.args[0] == "tracker"
    # check that the new message is present and the old one isn't
    assert "hello world" in json.dumps(call.args[1])
    assert "foo bar" not in json.dumps(call.args[1])

    retrieved_tracker = await agent_with_flows.tracker_store.retrieve(sender_id)
    assert retrieved_tracker is not None

    # old events should be gone from the tracker and there should only be
    # the new utterance stored on it
    assert len(retrieved_tracker.events) == 2
    assert isinstance(retrieved_tracker.events[0], UserUttered)
    assert retrieved_tracker.events[0].text == "hello world"
    assert isinstance(retrieved_tracker.events[1], ActionExecuted)
    assert retrieved_tracker.events[1].action_name == ACTION_LISTEN_NAME


async def test_inspector_handle_partial_tracker_update(
    inspector_input: InspectorInputChannel,
    default_tracker: DialogueStateTracker,
    agent_with_flows: Agent,
) -> None:
    sender_id = f"test_inspector_handle_partial_tracker_update_{uuid.uuid4().hex}"
    default_tracker.sender_id = sender_id
    default_tracker.update(UserUttered("foo bar"))
    await agent_with_flows.tracker_store.save(default_tracker)

    inspector_input.agent = agent_with_flows

    data = {
        "sender_id": sender_id,
        "events": [
            UserUttered("hello world").as_dict(),
        ],
    }
    await inspector_input.handle_tracker_update("some_sid", data)

    retrieved_tracker = await agent_with_flows.tracker_store.retrieve(sender_id)
    assert retrieved_tracker is not None

    # the conversation should have been continued and the last action should
    # be an action listen
    assert len(retrieved_tracker.events) > 0
    assert isinstance(retrieved_tracker.events[-1], ActionExecuted)
    assert retrieved_tracker.events[-1].action_name == ACTION_LISTEN_NAME


def test_inspector_from_credentials_parses_values() -> None:
    """Test that InspectorInputChannel parses credentials properly."""
    credentials = {
        "user_message_evt": "custom_user_msg",
        "bot_message_evt": "custom_bot_msg",
        "session_persistence": False,
        "server_url": "example.com",
        "asr": {"name": "test_asr"},
        "tts": {"name": "test_tts"},
        "voice_channel": "my_custom_voice",
    }
    input_channel = InspectorInputChannel.from_credentials(credentials)

    assert isinstance(input_channel, InspectorInputChannel)
    assert input_channel.user_message_evt == "custom_user_msg"
    assert input_channel.bot_message_evt == "custom_bot_msg"
    assert input_channel.session_persistence is False
    assert input_channel.server_url == "example.com"
    assert input_channel.asr_config == {"name": "test_asr"}
    assert input_channel.tts_config == {"name": "test_tts"}
    assert input_channel.voice_channel_name == "my_custom_voice"


def test_inspector_from_credentials_uses_defaults() -> None:
    """Test that defaults are used when credentials are None (rasa inspect)."""
    input_channel = InspectorInputChannel.from_credentials(None)

    assert isinstance(input_channel, InspectorInputChannel)
    assert input_channel.user_message_evt == "user_message"
    assert input_channel.bot_message_evt == "bot_message"
    assert input_channel.session_persistence is True
    assert input_channel.server_url == "localhost"
    assert input_channel.asr_config == {"name": "deepgram"}
    assert input_channel.tts_config == {"name": "deepgram"}
    assert input_channel.voice_channel_name == "browser_audio"
    assert input_channel.interruption_config.enabled is True
    assert input_channel.interruption_config.min_words == DEFAULT_INTERRUPTION_MIN_WORDS


def test_inspector_respects_explicit_interruptions_config() -> None:
    """Test that explicit interruptions config overrides the default."""
    credentials = {
        "interruptions": {"enabled": False},
    }
    input_channel = InspectorInputChannel.from_credentials(credentials)

    assert input_channel.interruption_config.enabled is False


def test_inspector_respects_custom_interruptions_min_words() -> None:
    """Test that custom min_words is respected in interruptions config."""
    credentials = {
        "interruptions": {"enabled": True, "min_words": 5},
    }
    input_channel = InspectorInputChannel.from_credentials(credentials)

    assert input_channel.interruption_config.enabled is True
    assert input_channel.interruption_config.min_words == 5


def test_inspector_channel_defaults_to_browser_audio() -> None:
    """Test that voice_channel defaults to browser_audio when not specified."""
    credentials = {"server_url": "localhost"}
    input_channel = InspectorInputChannel.from_credentials(credentials)
    assert input_channel.voice_channel_name == "browser_audio"


def test_inspector_channel_custom_name() -> None:
    """Test that voice_channel can be configured via credentials."""
    credentials = {"voice_channel": "my_voice"}
    input_channel = InspectorInputChannel.from_credentials(credentials)
    assert input_channel.voice_channel_name == "my_voice"


def test_inspector_text_channel_defaults_to_custom_text_channel() -> None:
    """Test that text_channel defaults to custom_text_channel when not specified."""
    credentials = {"server_url": "localhost"}
    input_channel = InspectorInputChannel.from_credentials(credentials)
    assert input_channel.text_channel_name == "custom_text_channel"


def test_inspector_text_channel_custom_name() -> None:
    """Test that text_channel can be configured via credentials."""
    credentials = {"text_channel": "my_text"}
    input_channel = InspectorInputChannel.from_credentials(credentials)
    assert input_channel.text_channel_name == "my_text"


def test_inspector_text_channel_default_when_no_credentials() -> None:
    """Test that text_channel defaults correctly when credentials are None."""
    input_channel = InspectorInputChannel.from_credentials(None)
    assert input_channel.text_channel_name == "custom_text_channel"


def test_inspector_get_output_channel_returns_text_output() -> None:
    """Test that get_output_channel returns InspectorTextOutputChannel."""
    sio = AsyncMock()
    channel = InspectorInputChannel()
    channel.sio_server = sio
    output = channel.get_output_channel()
    assert isinstance(output, InspectorTextOutputChannel)
    assert output.name() == "custom_text_channel"


def test_inspector_get_output_channel_uses_custom_name() -> None:
    """Test that get_output_channel uses a custom text_channel name."""
    sio = AsyncMock()
    channel = InspectorInputChannel(text_channel="my_text")
    channel.sio_server = sio
    output = channel.get_output_channel()
    assert isinstance(output, InspectorTextOutputChannel)
    assert output.name() == "my_text"


def test_inspector_get_output_channel_returns_none_without_sio() -> None:
    """Test that get_output_channel returns None when sio_server is None."""
    channel = InspectorInputChannel()
    assert channel.get_output_channel() is None


async def test_handle_voice_streaming_emits_voice_error_on_exception(
    inspector_input: InspectorInputChannel,
) -> None:
    """Test that _handle_voice_streaming emits voice_error when streaming fails."""
    sid = "test_sid"
    ws_adapter = AsyncMock()
    inspector_input.active_connections[sid] = ws_adapter

    error = RuntimeError("Missing environment variable for ASR Engine")

    with patch.object(
        inspector_input,
        "run_audio_streaming",
        new_callable=AsyncMock,
        side_effect=error,
    ):
        await inspector_input._handle_voice_streaming(AsyncMock(), ws_adapter, sid)

    inspector_input.sio_server.emit.assert_called_once_with(
        "voice_error",
        {
            "message": "Voice streaming failed",
            "error": str(error),
            "exception": "RuntimeError",
        },
        room=sid,
    )
    assert sid not in inspector_input.active_connections


async def test_handle_voice_streaming_no_emit_on_success(
    inspector_input: InspectorInputChannel,
) -> None:
    """Test that _handle_voice_streaming does not emit voice_error on success."""
    sid = "test_sid"
    ws_adapter = AsyncMock()

    with patch.object(inspector_input, "run_audio_streaming", new_callable=AsyncMock):
        await inspector_input._handle_voice_streaming(AsyncMock(), ws_adapter, sid)

    inspector_input.sio_server.emit.assert_not_called()


def test_inspector_from_credentials_parses_text_channel() -> None:
    """Test that from_credentials parses text_channel alongside other values."""
    credentials = {
        "voice_channel": "my_voice",
        "text_channel": "my_text",
        "server_url": "example.com",
    }
    input_channel = InspectorInputChannel.from_credentials(credentials)
    assert input_channel.voice_channel_name == "my_voice"
    assert input_channel.text_channel_name == "my_text"


# --- does_need_action_prediction ---


def test_does_need_action_prediction_empty_events(
    default_tracker: DialogueStateTracker,
) -> None:
    assert does_need_action_prediction(default_tracker) is True


def test_does_need_action_prediction_last_event_not_action(
    default_tracker: DialogueStateTracker,
) -> None:
    default_tracker.update(UserUttered("hello"))
    assert does_need_action_prediction(default_tracker) is True


def test_does_need_action_prediction_non_listen_action(
    default_tracker: DialogueStateTracker,
) -> None:
    default_tracker.update(ActionExecuted("action_greet"))
    assert does_need_action_prediction(default_tracker) is True


def test_does_need_action_prediction_action_listen(
    default_tracker: DialogueStateTracker,
) -> None:
    default_tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    assert does_need_action_prediction(default_tracker) is False


# --- InspectorTrackerUpdatePlugin ---


async def test_plugin_cancel_tasks(
    inspector_input: InspectorInputChannel,
) -> None:
    plugin = InspectorTrackerUpdatePlugin(inspector_input)

    async def long_running() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(long_running())
    plugin.tasks = [task]
    await plugin._cancel_tasks()

    assert plugin.tasks == []
    assert task.cancelled()


async def test_plugin_cleanup_tasks_removes_done(
    inspector_input: InspectorInputChannel,
) -> None:
    plugin = InspectorTrackerUpdatePlugin(inspector_input)

    async def immediate() -> None:
        return

    async def long_running() -> None:
        await asyncio.sleep(100)

    done_task = asyncio.create_task(immediate())
    await asyncio.sleep(0)
    pending_task = asyncio.create_task(long_running())
    plugin.tasks = [done_task, pending_task]

    plugin._cleanup_tasks()

    assert len(plugin.tasks) == 1
    assert plugin.tasks[0] is pending_task
    pending_task.cancel()
    await asyncio.gather(pending_task, return_exceptions=True)


async def test_plugin_after_response_chunk(
    inspector_input: InspectorInputChannel,
    default_tracker: DialogueStateTracker,
) -> None:
    plugin = InspectorTrackerUpdatePlugin(inspector_input)
    default_tracker.update(UserUttered("hi"))
    plugin.after_response_chunk(default_tracker, "streaming text")

    assert len(plugin.tasks) == 1
    await asyncio.sleep(0.1)

    inspector_input.sio_server.emit.assert_called()
    event_data = json.dumps(inspector_input.sio_server.emit.call_args.args[1])
    assert "streaming text" in event_data


async def test_plugin_after_server_stop(
    inspector_input: InspectorInputChannel,
) -> None:
    plugin = InspectorTrackerUpdatePlugin(inspector_input)

    async def long_running() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(long_running())
    plugin.tasks = [task]
    await plugin.after_server_stop()

    assert plugin.tasks == []
    assert task.cancelled()


# --- InspectorInputChannel: emit, emit_error, publish_streaming_response ---


async def test_emit_delegates_to_sio() -> None:
    channel = InspectorInputChannel()
    channel.sio_server = AsyncMock()
    await channel.emit("evt", {"k": "v"}, room="r1")
    channel.sio_server.emit.assert_called_once_with("evt", {"k": "v"}, room="r1")


async def test_emit_noop_without_sio() -> None:
    channel = InspectorInputChannel()
    channel.sio_server = None
    await channel.emit("evt", {"k": "v"}, room="r1")


async def test_emit_error(inspector_input: InspectorInputChannel) -> None:
    err = ValueError("broken")
    await inspector_input.emit_error("Oh no", "room1", err)

    call_args = inspector_input.sio_server.emit.call_args
    assert call_args.args[0] == "error"
    assert call_args.args[1]["message"] == "Oh no"
    assert call_args.args[1]["error"] == "broken"
    assert call_args.args[1]["exception"] == "ValueError"


async def test_publish_streaming_response(
    inspector_input: InspectorInputChannel,
    default_tracker: DialogueStateTracker,
) -> None:
    await inspector_input.publish_streaming_response(
        default_tracker.sender_id, default_tracker, "partial"
    )
    state = inspector_input.sio_server.emit.call_args.args[1]
    last_event = state["events"][-1]
    assert last_event["text"] == "partial"
    assert last_event["metadata"]["streaming"] is True


# --- InspectorInputChannel: on_message_proxy ---


async def test_on_message_proxy_success(
    inspector_input: InspectorInputChannel,
    default_tracker: DialogueStateTracker,
    default_agent: Agent,
) -> None:
    inspector_input.agent = default_agent
    default_tracker.sender_id = "test_on_message_proxy_success"
    default_tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    await default_agent.tracker_store.save(default_tracker)

    on_new_message = AsyncMock()
    msg = UserMessage("hello", sender_id=default_tracker.sender_id)
    await inspector_input.on_message_proxy(on_new_message, msg)

    on_new_message.assert_called_once_with(msg)
    inspector_input.sio_server.emit.assert_called()


async def test_on_message_proxy_agent_not_ready(
    inspector_input: InspectorInputChannel,
) -> None:
    inspector_input.agent = None
    msg = UserMessage("hello", sender_id="test_sender")
    await inspector_input.on_message_proxy(AsyncMock(), msg)

    assert any(
        c.args[0] == "error" for c in inspector_input.sio_server.emit.call_args_list
    )


# --- InspectorInputChannel: _cleanup_tasks_for_sid, after_server_stop ---


async def test_cleanup_tasks_for_sid() -> None:
    channel = InspectorInputChannel()
    channel.sio_server = AsyncMock()

    async def long_running() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(long_running())
    channel.background_tasks["sid1"] = task
    channel.active_connections["sid1"] = MagicMock()

    channel._cleanup_tasks_for_sid("sid1")

    assert "sid1" not in channel.background_tasks
    assert "sid1" not in channel.active_connections
    with pytest.raises(asyncio.CancelledError):
        await task


def test_cleanup_tasks_for_sid_noop_missing() -> None:
    channel = InspectorInputChannel()
    channel.sio_server = AsyncMock()
    channel._cleanup_tasks_for_sid("nonexistent")


async def test_channel_after_server_stop() -> None:
    channel = InspectorInputChannel()
    channel.sio_server = AsyncMock()

    async def long_running() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(long_running())
    channel.background_tasks["sid1"] = task
    channel.active_connections["sid1"] = MagicMock()

    await channel.after_server_stop()

    assert len(channel.background_tasks) == 0
    assert len(channel.active_connections) == 0
    assert task.cancelled()


# --- map_input_message ---


async def test_map_input_message_audio() -> None:
    channel = InspectorInputChannel()
    channel.sio_server = AsyncMock()
    raw = base64.b64encode(b"\x00\x01\x02\x03").decode()
    action = await channel.map_input_message({"audio": raw}, MagicMock())
    assert isinstance(action, NewAudioAction)


async def test_map_input_message_marker_match(_set_call_state) -> None:  # type: ignore[no-untyped-def]
    from rasa.core.channels.voice_stream.call_state import BotStoppedSpeaking

    channel = InspectorInputChannel()
    channel.sio_server = AsyncMock()
    _set_call_state.latest_bot_audio_id = "m1"
    _set_call_state.should_hangup = False

    action = await channel.map_input_message({"marker": "m1"}, MagicMock())
    assert isinstance(action, ContinueConversationAction)
    enqueued = _set_call_state.internal_queue.get_nowait()
    assert isinstance(enqueued, BotStoppedSpeaking)


async def test_map_input_message_marker_hangup(_set_call_state) -> None:  # type: ignore[no-untyped-def]
    channel = InspectorInputChannel()
    channel.sio_server = AsyncMock()
    _set_call_state.latest_bot_audio_id = "m1"
    _set_call_state.should_hangup = True

    action = await channel.map_input_message({"marker": "m1"}, MagicMock())
    assert isinstance(action, EndConversationAction)


async def test_map_input_message_marker_non_matching(_set_call_state) -> None:  # type: ignore[no-untyped-def]
    from rasa.core.channels.voice_stream.call_state import BotIsSpeaking

    channel = InspectorInputChannel()
    channel.sio_server = AsyncMock()
    _set_call_state.latest_bot_audio_id = "other"

    action = await channel.map_input_message({"marker": "xyz"}, MagicMock())
    assert isinstance(action, ContinueConversationAction)
    enqueued = _set_call_state.internal_queue.get_nowait()
    assert isinstance(enqueued, BotIsSpeaking)


async def test_map_input_message_unknown() -> None:
    channel = InspectorInputChannel()
    channel.sio_server = AsyncMock()
    action = await channel.map_input_message({"something": True}, MagicMock())
    assert isinstance(action, ContinueConversationAction)


# --- SocketIOVoiceWebsocketAdapter ---


async def test_adapter_send() -> None:
    sio = AsyncMock()
    adapter = SocketIOVoiceWebsocketAdapter(sio, "sess", "sid1", "bot_msg")
    await adapter.send({"text": "hi"})
    sio.emit.assert_called_once_with("bot_msg", {"text": "hi"}, room="sid1")


async def test_adapter_send_noop_when_closed() -> None:
    sio = AsyncMock()
    adapter = SocketIOVoiceWebsocketAdapter(sio, "sess", "sid1", "bot_msg")
    await adapter.close()
    await adapter.send({"text": "hi"})
    sio.emit.assert_not_called()


async def test_adapter_recv() -> None:
    sio = AsyncMock()
    adapter = SocketIOVoiceWebsocketAdapter(sio, "sess", "sid1", "bot_msg")
    adapter.put_message({"audio": "data"})
    assert await adapter.recv() == {"audio": "data"}


async def test_adapter_recv_closed() -> None:
    sio = AsyncMock()
    adapter = SocketIOVoiceWebsocketAdapter(sio, "sess", "sid1", "bot_msg")
    await adapter.close()
    with pytest.raises(ConnectionError):
        await adapter.recv()


async def test_adapter_closed_property() -> None:
    adapter = SocketIOVoiceWebsocketAdapter(AsyncMock(), "s", "sid", "evt")
    assert adapter.closed is False
    await adapter.close()
    assert adapter.closed is True


async def test_adapter_aiter() -> None:
    adapter = SocketIOVoiceWebsocketAdapter(AsyncMock(), "s", "sid", "evt")
    adapter.put_message("a")
    adapter.put_message("b")

    received = []
    async for msg in adapter:
        received.append(msg)
        if len(received) >= 2:
            await adapter.close()

    assert received == ["a", "b"]


async def test_adapter_anext_stopped_when_closed() -> None:
    adapter = SocketIOVoiceWebsocketAdapter(AsyncMock(), "s", "sid", "evt")
    await adapter.close()
    with pytest.raises(StopAsyncIteration):
        await adapter.__anext__()


# --- InspectorVoiceOutputChannel ---


def test_voice_output_name() -> None:
    ch = InspectorVoiceOutputChannel(MagicMock(), MagicMock(), None, None, "custom")
    assert ch.name() == "custom"


def test_voice_output_default_name() -> None:
    ch = InspectorVoiceOutputChannel(MagicMock(), MagicMock(), None, None)
    assert ch.name() == "browser_audio"


def test_voice_output_bytes_to_message() -> None:
    ch = InspectorVoiceOutputChannel(MagicMock(), MagicMock(), None, None)
    raw = b"\x00\x01\x02\x03"
    parsed = json.loads(ch.channel_bytes_to_message("r", raw))
    assert base64.b64decode(parsed["audio"]) == raw


def test_voice_output_marker_no_latency(_set_call_state) -> None:  # type: ignore[no-untyped-def]
    _set_call_state.asr_latency_ms = None
    _set_call_state.rasa_processing_latency_ms = None
    _set_call_state.tts_first_byte_latency_ms = None
    _set_call_state.tts_complete_latency_ms = None

    ch = InspectorVoiceOutputChannel(MagicMock(), MagicMock(), None, None)
    marker_json, _ = ch.create_marker_message("r")
    parsed = json.loads(marker_json)
    assert "marker" in parsed
    assert "latency" not in parsed


def test_voice_output_marker_with_latency(_set_call_state) -> None:  # type: ignore[no-untyped-def]
    _set_call_state.asr_latency_ms = 100
    _set_call_state.rasa_processing_latency_ms = 200
    _set_call_state.tts_first_byte_latency_ms = None
    _set_call_state.tts_complete_latency_ms = 50

    ch = InspectorVoiceOutputChannel(MagicMock(), MagicMock(), None, None)
    marker_json, mid = ch.create_marker_message("r")
    parsed = json.loads(marker_json)
    assert parsed["latency"]["asr_latency_ms"] == 100
    assert parsed["latency"]["rasa_processing_latency_ms"] == 200
    assert parsed["latency"]["tts_complete_latency_ms"] == 50
    assert "tts_first_byte_latency_ms" not in parsed["latency"]
