import pytest

import rasa.utils.io
from rasa.core import restore
from rasa.core.agent import Agent


@pytest.fixture(scope="module")
def loop():
    from pytest_sanic.plugin import loop as sanic_loop

    return rasa.utils.io.enable_async_loop_debugging(next(sanic_loop()))


async def test_restoring_tracker(trained_moodbot_path, recwarn):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"

    agent = Agent.load(trained_moodbot_path)

    tracker = restore.load_tracker_from_json(tracker_dump, agent.domain)

    await restore.replay_events(tracker, agent)

    # makes sure there are no warnings. warnings are raised, if the models
    # predictions differ from the tracker when the dumped tracker is replayed
    assert [e for e in recwarn if e._category_name == "UserWarning"] == []

    assert len(tracker.events) == 7
    assert tracker.latest_action_name == "action_listen"
    assert not tracker.is_paused()
    assert tracker.sender_id == "mysender"
    assert tracker.events[-1].timestamp == 1517821726.211042
