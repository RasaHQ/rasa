from rasa.core.actions.action_repeat_bot_messages import ActionRepeatBotMessages
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.slots import FloatSlot, TextSlot
from rasa.shared.core.trackers import DialogueStateTracker


async def test_repeat_bot_utterance_with_collect_step():
    user1, bot1, user2, bot3, bot4, user3 = [
        UserUttered("search hotels"),
        BotUttered("How many rooms?"),
        UserUttered("Check how much money I have"),
        BotUttered("You have 1000 dollars"),
        BotUttered("How many rooms?"),
        UserUttered("Sorry, come again?"),
    ]

    stack = DialogueStack.from_dict(
        [
            {
                "flow_id": "search_hotels",
                "frame_id": "OGE1U359",
                "frame_type": "regular",
                "step_id": "1_collect_num_rooms",
                "type": "flow",
            },
            {
                "collect": "num_rooms",
                "flow_id": "pattern_collect_information",
                "frame_id": "39LEDJUN",
                "rejections": [],
                "step_id": "listen",
                "type": "pattern_collect_information",
                "utter": "utter_ask_num_rooms",
                "collect_action": "action_ask_num_rooms",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        [user1, bot1, user2, bot3, bot4, user3],
        slots=[
            FloatSlot("num_rooms", mappings=[]),
            TextSlot("start_date", mappings=[]),
            TextSlot("end_slot", mappings=[]),
        ],
    )
    tracker.update_stack(stack)

    domain = Domain.empty()
    action = ActionRepeatBotMessages()
    channel = CollectingOutputChannel()
    nlg = NaturalLanguageGenerator()
    events = await action.run(channel, nlg, tracker, domain)
    assert events == [bot3]


async def test_repeat_bot_utterance():
    user1, bot1, user2, bot3, bot4, user3 = [
        UserUttered("search hotels"),
        BotUttered("How many rooms?"),
        UserUttered("Check how much money I have"),
        BotUttered("You have 1000 dollars"),
        BotUttered("How many rooms?"),
        UserUttered("Sorry, come again?"),
    ]

    stack = DialogueStack.from_dict([])
    tracker = DialogueStateTracker.from_events(
        "test",
        [user1, bot1, user2, bot3, bot4, user3],
        slots=[
            FloatSlot("num_rooms", mappings=[]),
            TextSlot("start_date", mappings=[]),
            TextSlot("end_slot", mappings=[]),
        ],
    )
    tracker.update_stack(stack)

    domain = Domain.empty()
    action = ActionRepeatBotMessages()
    channel = CollectingOutputChannel()
    nlg = NaturalLanguageGenerator()
    events = await action.run(channel, nlg, tracker, domain)
    assert events == [bot3, bot4]
