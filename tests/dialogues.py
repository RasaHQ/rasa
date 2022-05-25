from rasa.shared.core.conversation import Dialogue
from rasa.shared.core.events import (
    SlotSet,
    UserUttered,
    ActionExecuted,
    ActiveLoop,
    BotUttered,
)


TEST_DEFAULT_DIALOGUE = Dialogue(
    name="default",
    events=[
        ActionExecuted(action_name="action_listen", timestamp=1551952977.4850519),
        UserUttered(
            entities=[{"end": 19, "entity": "name", "start": 14, "value": "Peter"}],
            intent={"confidence": 0.0, "name": "greet"},
            message_id=None,
            parse_data={
                "entities": [
                    {"end": 19, "entity": "name", "start": 14, "value": "Peter"}
                ],
                "intent": {"confidence": 0.0, "name": "greet"},
                "message_id": None,
                "metadata": {},
                "text": "Hi my name is Peter",
            },
            text="Hi my name is Peter",
            timestamp=1551953035.076376,
        ),
        SlotSet(key="name", timestamp=1551953035.076385, value="Peter"),
        ActionExecuted(action_name="utter_greet", timestamp=1551953040.607782),
        BotUttered(
            data={"attachment": None, "buttons": None, "elements": None},
            text="hey there Peter!",
            timestamp=1551953040.60779,
        ),
    ],
)
TEST_FORMBOT_DIALOGUE = Dialogue(
    name="formbot",
    events=[
        ActionExecuted(action_name="action_listen", timestamp=1551884035.892855),
        UserUttered(
            intent={"confidence": 0.3748943507671356, "name": "greet"},
            parse_data={
                "entities": [],
                "intent": {"confidence": 0.3748943507671356, "name": "greet"},
                "text": "Hi I'm desperate to talk to you",
            },
            text="Hi I'm desperate to talk to you",
            timestamp=1551884050.259948,
        ),
        ActionExecuted(
            action_name="utter_greet",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551884060.466681,
        ),
        BotUttered(
            data={"attachment": None, "buttons": None, "elements": None},
            text="Hello! I am restaurant search assistant! How can I help?",
            timestamp=1551884060.46669,
        ),
        ActionExecuted(
            action_name="action_listen",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551884061.9350882,
        ),
        UserUttered(
            entities=[
                {"end": 18, "entity": "cuisine", "start": 16, "value": "an"},
                {"end": 48, "entity": "location", "start": 42, "value": "Bombay"},
            ],
            intent={"confidence": 0.9414282441139221, "name": "request_restaurant"},
            parse_data={
                "entities": [
                    {"end": 18, "entity": "cuisine", "start": 16, "value": "an"},
                    {"end": 48, "entity": "location", "start": 42, "value": "Bombay"},
                ],
                "intent": {
                    "confidence": 0.9414282441139221,
                    "name": "request_restaurant",
                },
                "text": "I'm looking for an indian restaurant...in Bombay",
            },
            text="I'm looking for an indian restaurant...in Bombay",
            timestamp=1551884090.9653602,
        ),
        ActionExecuted(
            action_name="restaurant_form",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551884095.542748,
        ),
        ActionExecuted(
            action_name="utter_slots_values",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551884097.570883,
        ),
        BotUttered(
            data={"attachment": None, "buttons": None, "elements": None},
            text=(
                "I am going to run a restaurant search "
                "using the following parameters:\n"
                " - cuisine: None\n - num_people: None\n"
                " - outdoor_seating: None\n"
                " - preferences: None\n - feedback: None"
            ),
            timestamp=1551884097.57089,
        ),
        ActionExecuted(
            action_name="action_listen",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551884098.8006358,
        ),
        UserUttered(
            intent={"confidence": 0.2287036031484604, "name": "affirm"},
            parse_data={
                "entities": [],
                "intent": {"confidence": 0.2287036031484604, "name": "affirm"},
                "text": "Let's just pretend everything went correctly",
            },
            text="Let's just pretend everything went correctly",
            timestamp=1551884208.092693,
        ),
        ActionExecuted(
            action_name="action_deactivate_loop", timestamp=1551884214.951055
        ),
        ActiveLoop(name=None, timestamp=1551884214.9510589),
        SlotSet(key="requested_slot", timestamp=1551884214.951062, value=None),
        ActionExecuted(
            action_name="action_listen",
            confidence=0.7680902069097734,
            policy="policy_0_TEDPolicy",
            timestamp=1551884216.705635,
        ),
    ],
)
TEST_MOODBOT_DIALOGUE = Dialogue(
    name="moodbot",
    events=[
        ActionExecuted(action_name="action_listen", timestamp=1551883958.346432),
        UserUttered(
            intent={"confidence": 0.44488201660555066, "name": "greet"},
            parse_data={
                "entities": [],
                "intent": {"confidence": 0.44488201660555066, "name": "greet"},
                "intent_ranking": [
                    {"confidence": 0.44488201660555066, "name": "greet"},
                    {"confidence": 0.29023286595689257, "name": "goodbye"},
                    {"confidence": 0.10501227521380094, "name": "mood_great"},
                    {"confidence": 0.06879303900502878, "name": "mood_unhappy"},
                    {"confidence": 0.04903582960375451, "name": "deny"},
                    {"confidence": 0.04204397361497238, "name": "affirm"},
                ],
                "text": "Hi talk to me",
            },
            text="Hi talk to me",
            timestamp=1551883971.410778,
        ),
        ActionExecuted(
            action_name="utter_greet",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551883975.6456478,
        ),
        BotUttered(
            data={
                "attachment": None,
                "buttons": [
                    {"payload": "great", "title": "great"},
                    {"payload": "super sad", "title": "super sad"},
                ],
                "elements": None,
            },
            text="Hey! How are you?",
            timestamp=1551883975.645656,
        ),
        ActionExecuted(
            action_name="action_listen",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551883979.098331,
        ),
        UserUttered(
            intent={"confidence": 0.7417646502470048, "name": "mood_unhappy"},
            parse_data={
                "entities": [],
                "intent": {"confidence": 0.7417646502470048, "name": "mood_unhappy"},
                "intent_ranking": [
                    {"confidence": 0.7417646502470048, "name": "mood_unhappy"},
                    {"confidence": 0.1439688162980615, "name": "mood_great"},
                    {"confidence": 0.04577343822867981, "name": "goodbye"},
                    {"confidence": 0.037760394267609965, "name": "greet"},
                    {"confidence": 0.017715563733253295, "name": "affirm"},
                    {"confidence": 0.013017137225390567, "name": "deny"},
                ],
                "text": "Super sad",
            },
            text="Super sad",
            timestamp=1551883982.540276,
        ),
        ActionExecuted(
            action_name="utter_cheer_up",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551883985.031668,
        ),
        BotUttered(
            data={
                "attachment": "https://i.imgur.com/nGF1K8f.jpg",
                "buttons": None,
                "elements": None,
            },
            text="Here is something to cheer you up:",
            timestamp=1551883985.0316749,
        ),
        ActionExecuted(
            action_name="utter_did_that_help",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551883985.940413,
        ),
        BotUttered(
            data={"attachment": None, "buttons": None, "elements": None},
            text="Did that help you?",
            timestamp=1551883985.940421,
        ),
        ActionExecuted(
            action_name="action_listen",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551883986.958556,
        ),
        UserUttered(
            intent={"confidence": 0.8162296627642036, "name": "deny"},
            parse_data={
                "entities": [],
                "intent": {"confidence": 0.8162296627642036, "name": "deny"},
                "intent_ranking": [
                    {"confidence": 0.8162296627642036, "name": "deny"},
                    {"confidence": 0.07152463661481759, "name": "mood_unhappy"},
                    {"confidence": 0.05028159510181415, "name": "greet"},
                    {"confidence": 0.02662414324721426, "name": "affirm"},
                    {"confidence": 0.024343883584915963, "name": "goodbye"},
                    {"confidence": 0.010996078687034375, "name": "mood_great"},
                ],
                "text": "No",
            },
            text="No",
            timestamp=1551883989.0720608,
        ),
        ActionExecuted(
            action_name="utter_goodbye",
            confidence=1.0,
            policy="policy_2_MemoizationPolicy",
            timestamp=1551883991.061463,
        ),
        BotUttered(
            data={"attachment": None, "buttons": None, "elements": None},
            text="Bye",
            timestamp=1551883991.061471,
        ),
    ],
)

TEST_DIALOGUES = [TEST_DEFAULT_DIALOGUE, TEST_FORMBOT_DIALOGUE, TEST_MOODBOT_DIALOGUE]

TEST_DOMAINS_FOR_DIALOGUES = [
    "data/test_domains/default_with_slots.yml",
    "examples/formbot/domain.yml",
    "data/test_moodbot/domain.yml",
]
