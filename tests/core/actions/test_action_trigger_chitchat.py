from rasa.core.actions.action_trigger_chitchat import ActionTriggerChitchat
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.stack.frames import ChitChatStackFrame
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.trackers import DialogueStateTracker


async def test_action_trigger_chitchat():
    tracker = DialogueStateTracker.from_events("test", [])
    action = ActionTriggerChitchat()
    channel = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator({})
    events = await action.run(channel, nlg, tracker, Domain.empty())
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert isinstance(frame, ChitChatStackFrame)
