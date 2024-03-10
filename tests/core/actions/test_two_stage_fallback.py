from typing import List, Text

import pytest

from rasa.core.policies.policy import PolicyPrediction
from rasa.core.processor import MessageProcessor
from rasa.shared.constants import (
    DEFAULT_NLU_FALLBACK_INTENT_NAME,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
)
from rasa.core.actions.two_stage_fallback import TwoStageFallbackAction
from rasa.core.channels import CollectingOutputChannel
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    SlotSet,
    ActiveLoop,
    BotUttered,
    UserUtteranceReverted,
    Event,
)
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import INTENT_RANKING_KEY
from rasa.shared.core.constants import (
    USER_INTENT_OUT_OF_SCOPE,
    ACTION_LISTEN_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
)


def _message_requiring_fallback() -> List[Event]:
    return [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(
            "hi",
            {"name": DEFAULT_NLU_FALLBACK_INTENT_NAME},
            parse_data={
                INTENT_RANKING_KEY: [
                    {"name": DEFAULT_NLU_FALLBACK_INTENT_NAME},
                    {"name": "greet"},
                    {"name": "bye"},
                ]
            },
        ),
    ]


def _two_stage_clarification_request() -> List[Event]:
    return [ActionExecuted(ACTION_TWO_STAGE_FALLBACK_NAME), BotUttered("please affirm")]


@pytest.mark.parametrize(
    "events",
    [
        _message_requiring_fallback(),
        _message_requiring_fallback()
        + [UserUtteranceReverted()]
        + _message_requiring_fallback(),
    ],
)
async def test_ask_affirmation(events: List[Event]):
    tracker = DialogueStateTracker.from_events("some-sender", evts=events)
    domain = Domain.empty()
    action = TwoStageFallbackAction()

    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert len(events) == 2
    assert events[0] == ActiveLoop(ACTION_TWO_STAGE_FALLBACK_NAME)
    assert isinstance(events[1], BotUttered)


async def test_1st_affirmation_is_successful(default_processor: MessageProcessor):
    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("my name is John", {"name": "say_name", "confidence": 1.0}),
            SlotSet("some_slot", "example_value"),
            # User sends message with low NLU confidence
            *_message_requiring_fallback(),
            ActiveLoop(ACTION_TWO_STAGE_FALLBACK_NAME),
            # Action asks user to affirm
            *_two_stage_clarification_request(),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User affirms
            UserUttered("hi", {"name": "greet", "confidence": 1.0}),
        ],
    )
    domain = Domain.empty()
    action = TwoStageFallbackAction()

    await default_processor._run_action(
        action,
        tracker,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        PolicyPrediction([], "some policy"),
    )

    applied_events = tracker.applied_events()
    assert applied_events == [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("my name is John", {"name": "say_name", "confidence": 1.0}),
        SlotSet("some_slot", "example_value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("hi", {"name": "greet", "confidence": 1.0}),
    ]


async def test_give_it_up_after_low_confidence_after_affirm_request():
    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            # User sends message with low NLU confidence
            *_message_requiring_fallback(),
            ActiveLoop(ACTION_TWO_STAGE_FALLBACK_NAME),
            # Action asks user to affirm
            *_two_stage_clarification_request(),
            # User's affirms with low NLU confidence again
            *_message_requiring_fallback(),
        ],
    )
    domain = Domain.empty()
    action = TwoStageFallbackAction()

    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert events == [ActiveLoop(None), UserUtteranceReverted()]


async def test_ask_rephrase_after_failed_affirmation():
    rephrase_text = "please rephrase"
    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            # User sends message with low NLU confidence
            *_message_requiring_fallback(),
            ActiveLoop(ACTION_TWO_STAGE_FALLBACK_NAME),
            # Action asks user to affirm
            *_two_stage_clarification_request(),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User denies suggested intents
            UserUttered("hi", {"name": USER_INTENT_OUT_OF_SCOPE}),
        ],
    )

    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_ask_rephrase:
            - text: {rephrase_text}
        """
    )
    action = TwoStageFallbackAction()

    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert len(events) == 1
    assert isinstance(events[0], BotUttered)

    bot_utterance = events[0]
    assert isinstance(bot_utterance, BotUttered)
    assert bot_utterance.text == rephrase_text


async def test_ask_rephrasing_successful(default_processor: MessageProcessor):
    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("my name is John", {"name": "say_name", "confidence": 1.0}),
            SlotSet("some_slot", "example_value"),
            # User sends message with low NLU confidence
            *_message_requiring_fallback(),
            ActiveLoop(ACTION_TWO_STAGE_FALLBACK_NAME),
            # Action asks user to affirm
            *_two_stage_clarification_request(),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User denies suggested intents
            UserUttered("hi", {"name": USER_INTENT_OUT_OF_SCOPE}),
            *_two_stage_clarification_request(),
            # Action asks user to rephrase
            ActionExecuted(ACTION_LISTEN_NAME),
            # User rephrases successfully
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    domain = Domain.empty()
    action = TwoStageFallbackAction()

    await default_processor._run_action(
        action,
        tracker,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        PolicyPrediction([], "some policy"),
    )

    applied_events = tracker.applied_events()
    assert applied_events == [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("my name is John", {"name": "say_name", "confidence": 1.0}),
        SlotSet("some_slot", "example_value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("hi", {"name": "greet"}),
    ]


async def test_ask_affirm_after_rephrasing():
    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            # User sends message with low NLU confidence
            *_message_requiring_fallback(),
            ActiveLoop(ACTION_TWO_STAGE_FALLBACK_NAME),
            # Action asks user to affirm
            *_two_stage_clarification_request(),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User denies suggested intents
            UserUttered("hi", {"name": USER_INTENT_OUT_OF_SCOPE}),
            # Action asks user to rephrase
            ActionExecuted(ACTION_TWO_STAGE_FALLBACK_NAME),
            BotUttered("please rephrase"),
            # User rephrased with low confidence
            *_message_requiring_fallback(),
        ],
    )
    domain = Domain.empty()
    action = TwoStageFallbackAction()

    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert len(events) == 1
    assert isinstance(events[0], BotUttered)


async def test_2nd_affirm_successful(default_processor: MessageProcessor):
    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("my name is John", {"name": "say_name", "confidence": 1.0}),
            SlotSet("some_slot", "example_value"),
            # User sends message with low NLU confidence
            *_message_requiring_fallback(),
            ActiveLoop(ACTION_TWO_STAGE_FALLBACK_NAME),
            # Action asks user to affirm
            *_two_stage_clarification_request(),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User denies suggested intents
            UserUttered("hi", {"name": USER_INTENT_OUT_OF_SCOPE}),
            # Action asks user to rephrase
            *_two_stage_clarification_request(),
            # User rephrased with low confidence
            *_message_requiring_fallback(),
            *_two_stage_clarification_request(),
            # Actions asks user to affirm for the last time
            ActionExecuted(ACTION_LISTEN_NAME),
            # User affirms successfully
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    domain = Domain.empty()
    action = TwoStageFallbackAction()

    await default_processor._run_action(
        action,
        tracker,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        PolicyPrediction([], "some policy"),
    )

    applied_events = tracker.applied_events()

    assert applied_events == [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("my name is John", {"name": "say_name", "confidence": 1.0}),
        SlotSet("some_slot", "example_value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("hi", {"name": "greet"}),
    ]


@pytest.mark.parametrize(
    "intent_which_lets_action_give_up",
    [USER_INTENT_OUT_OF_SCOPE, DEFAULT_NLU_FALLBACK_INTENT_NAME],
)
async def test_2nd_affirmation_failed(intent_which_lets_action_give_up: Text):
    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            # User sends message with low NLU confidence
            *_message_requiring_fallback(),
            ActiveLoop(ACTION_TWO_STAGE_FALLBACK_NAME),
            # Action asks user to affirm
            *_two_stage_clarification_request(),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User denies suggested intents
            UserUttered("hi", {"name": USER_INTENT_OUT_OF_SCOPE}),
            # Action asks user to rephrase
            *_two_stage_clarification_request(),
            # User rephrased with low confidence
            *_message_requiring_fallback(),
            # Actions asks user to affirm for the last time
            *_two_stage_clarification_request(),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User denies suggested intents for the second time
            UserUttered("hi", {"name": intent_which_lets_action_give_up}),
        ],
    )
    domain = Domain.empty()
    action = TwoStageFallbackAction()

    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert events == [ActiveLoop(None), UserUtteranceReverted()]
