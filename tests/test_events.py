from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from copy import deepcopy

import pytest

from rasa_core.events import UserUttered, TopicSet, SlotSet, Restarted, \
    ActionExecuted, AllSlotsReset, \
    ReminderScheduled, ConversationResumed, ConversationPaused, StoryExported, \
    ActionReverted


@pytest.mark.parametrize("one_event,another_event", [
    (UserUttered("_greet", "greet", []),
     UserUttered("_goodbye", "goodbye", [])),

    (TopicSet("my_topic"),
     TopicSet("my_other_topic")),

    (SlotSet("my_slot", "value"),
     SlotSet("my__other_slot", "value")),

    (Restarted(),
     None),

    (AllSlotsReset(),
     None),

    (ConversationPaused(),
     None),

    (ConversationResumed(),
     None),

    (StoryExported(),
     None),

    (ActionReverted(),
     None),

    (ActionExecuted("my_action"),
     ActionExecuted("my_other_action")),

    (ReminderScheduled("my_action", "now"),
     ReminderScheduled("my_other_action", "now")),
])
def test_event_has_proper_implementation(one_event, another_event):
    # equals tests
    assert one_event != another_event, \
        "Same events with different values need to be different"
    assert one_event == deepcopy(one_event), \
        "Event copies need to be the same"
    assert one_event != 42, \
        "Events aren't equal to 42!"

    # hash test
    assert hash(one_event) == hash(deepcopy(one_event)), \
        "Same events should have the same hash"
    assert hash(one_event) != hash(another_event), \
        "Different events should have different hashes"

    # str test
    assert "object at 0x" not in str(one_event), \
        "Event has a proper str method"
