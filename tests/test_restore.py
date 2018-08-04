from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core import restore


def test_restoring_tracker(trained_moodbot_path, recwarn):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"

    agent, tracker = restore.recreate_agent(trained_moodbot_path,
                                            tracker_dump=tracker_dump)

    # makes sure there are no warnings. warnings are raised, if the models
    # predictions differ from the tracker when the dumped tracker is replayed
    assert [e
            for e in recwarn
            if e._category_name == "UserWarning"] == []

    assert len(tracker.events) == 7
    assert tracker.latest_action_name == "action_listen"
    assert not tracker.is_paused()
    assert tracker.sender_id == "mysender"
    assert tracker.events[-1].timestamp == 1517821726.211042
