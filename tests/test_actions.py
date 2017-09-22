from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.events import Restarted

from rasa_core.actions.action import ActionRestart
from rasa_core.trackers import DialogueStateTracker


def test_restart(default_dispatcher, default_domain):
    tracker = DialogueStateTracker("default",
                                   default_domain.slots,
                                   default_domain.topics,
                                   default_domain.default_topic)
    events = ActionRestart().run(default_dispatcher, tracker, default_domain)
    assert events == [Restarted()]
