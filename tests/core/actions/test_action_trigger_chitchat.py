from rasa.core.actions.action_trigger_chitchat import ActionTriggerChitchat
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.stack.frames import ChitChatStackFrame
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.trackers import DialogueStateTracker


async def test_action_trigger_chitchat():
    tracker = DialogueStateTracker.from_events("test", [])
    action = ActionTriggerChitchat()
    channel = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator({})
    events = await action.run(channel, nlg, tracker, Domain.empty())
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SlotSet)
    assert event.key == DIALOGUE_STACK_SLOT
    assert len(event.value) == 1
    assert event.value[0]["type"] == ChitChatStackFrame.type()
