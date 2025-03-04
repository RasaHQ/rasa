import asyncio
import json
from unittest.mock import AsyncMock, call

import pytest

from rasa.core.agent import Agent
from rasa.core.channels.studio_chat import (
    StudioChatInput,
    StudioTrackerUpdatePlugin,
    tracker_as_dump,
)
from rasa.shared.core.constants import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME
from rasa.shared.core.events import ActionExecuted, SessionStarted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.fixture
def studio_input() -> StudioChatInput:
    sio = AsyncMock()
    output_channel = StudioChatInput()
    output_channel.sio = sio
    return output_channel


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

    dump = tracker_as_dump(default_tracker)
    loaded = json.loads(dump)

    # there should be one action listen and one action session start
    assert len(loaded.get("events")) == 2


async def test_tracker_update_plugin_triggers_after_new_user_message(
    studio_input: StudioChatInput,
    default_tracker: DialogueStateTracker,
) -> None:
    default_tracker.update(UserUttered("hello world"))

    plugin = StudioTrackerUpdatePlugin(studio_input)
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
    studio_input.sio.emit.assert_has_calls(expected_calls, any_order=False)


async def test_tracker_update_plugin_triggers_after_action_executed(
    studio_input: StudioChatInput,
    default_tracker: DialogueStateTracker,
) -> None:
    default_tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    plugin = StudioTrackerUpdatePlugin(studio_input)
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
    studio_input.sio.emit.assert_has_calls(expected_calls, any_order=False)


async def test_studio_chat_handle_tracker_update(
    studio_input: StudioChatInput,
    default_tracker: DialogueStateTracker,
    default_agent: Agent,
) -> None:
    default_tracker.sender_id = "test_studio_chat_handle_tracker_update"
    default_tracker.update(UserUttered("foo bar"))
    await default_agent.tracker_store.save(default_tracker)

    studio_input.agent = default_agent

    data = {
        "sender_id": default_tracker.sender_id,
        "events": [
            UserUttered("hello world").as_dict(),
            ActionExecuted(ACTION_LISTEN_NAME).as_dict(),
        ],
    }
    await studio_input.handle_tracker_update("some_sid", data)

    assert len(studio_input.sio.emit.call_args_list) == 1
    call = studio_input.sio.emit.call_args_list[0]
    assert call.args[0] == "tracker"
    # check that the new message is present and the old one isn't
    assert "hello world" in call.args[1]
    assert "foo bar" not in call.args[1]

    retrieved_tracker = await default_agent.tracker_store.retrieve(
        default_tracker.sender_id
    )
    assert retrieved_tracker is not None

    # old events should be gone from the tracker and there should only be
    # the new utterance stored on it
    assert len(retrieved_tracker.events) == 2
    assert isinstance(retrieved_tracker.events[0], UserUttered)
    assert retrieved_tracker.events[0].text == "hello world"
    assert isinstance(retrieved_tracker.events[1], ActionExecuted)
    assert retrieved_tracker.events[1].action_name == ACTION_LISTEN_NAME


async def test_studio_chat_handle_partial_tracker_update(
    studio_input: StudioChatInput,
    default_tracker: DialogueStateTracker,
    default_agent: Agent,
) -> None:
    default_tracker.sender_id = "test_studio_chat_handle_partial_tracker_update"
    default_tracker.update(UserUttered("foo bar"))
    await default_agent.tracker_store.save(default_tracker)

    studio_input.agent = default_agent

    data = {
        "sender_id": default_tracker.sender_id,
        "events": [
            UserUttered("hello world").as_dict(),
        ],
    }
    await studio_input.handle_tracker_update("some_sid", data)

    retrieved_tracker = await default_agent.tracker_store.retrieve(
        default_tracker.sender_id
    )
    assert retrieved_tracker is not None

    # the conversation should have been continued and the last action should
    # be an action listen
    assert len(retrieved_tracker.events) > 0
    assert isinstance(retrieved_tracker.events[-1], ActionExecuted)
    assert retrieved_tracker.events[-1].action_name == ACTION_LISTEN_NAME
