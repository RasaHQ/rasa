from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from datetime import datetime
import copy

import pytest

from rasa_core.events import (
    Event, UserUttered, SlotSet, Restarted,
    ActionExecuted, AllSlotsReset,
    ReminderScheduled, ConversationResumed, ConversationPaused,
    StoryExported, ActionReverted, BotUttered)


@pytest.mark.parametrize("one_event,another_event", [
    (UserUttered("/greet", {"name": "greet", "confidence": 1.0}, []),
     UserUttered("/goodbye", {"name": "goodbye", "confidence": 1.0}, [])),

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

    (BotUttered("my_text", "my_data"),
     BotUttered("my_other_test", "my_other_data")),

    (ReminderScheduled("my_action", "now"),
     ReminderScheduled("my_other_action", "now")),
])
def test_event_has_proper_implementation(one_event, another_event):
    # equals tests
    assert one_event != another_event, \
        "Same events with different values need to be different"
    assert one_event == copy.deepcopy(one_event), \
        "Event copies need to be the same"
    assert one_event != 42, \
        "Events aren't equal to 42!"

    # hash test
    assert hash(one_event) == hash(copy.deepcopy(one_event)), \
        "Same events should have the same hash"
    assert hash(one_event) != hash(another_event), \
        "Different events should have different hashes"

    # str test
    assert "object at 0x" not in str(one_event), \
        "Event has a proper str method"


@pytest.mark.parametrize("one_event", [
    UserUttered("/greet", {"name": "greet", "confidence": 1.0}, []),

    SlotSet("name", "rasa"),

    Restarted(),

    AllSlotsReset(),

    ConversationPaused(),

    ConversationResumed(),

    StoryExported(),

    ActionReverted(),

    ActionExecuted("my_action"),

    BotUttered("my_text", "my_data"),

    ReminderScheduled("my_action", datetime.now())
])
def test_dict_serialisation(one_event):
    evt_dict = one_event.as_dict()
    recovered_event = Event.from_parameters(evt_dict)
    assert hash(one_event) == hash(recovered_event)
