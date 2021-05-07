from pathlib import Path
from typing import Text
import pytest
from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.core import training
from rasa.core.actions.action import ActionDefaultFallback
from rasa.core.channels import CollectingOutputChannel
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_BACK_NAME,
    RULE_SNIPPET_ACTION_NAME,
    REQUESTED_SLOT,
    USER,
    PREVIOUS_ACTION,
    ACTIVE_LOOP,
    LOOP_NAME,
    RULE_ONLY_SLOTS,
    RULE_ONLY_LOOPS,
)
from rasa.shared.nlu.constants import TEXT, INTENT, ACTION_NAME, ENTITY_ATTRIBUTE_TYPE
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    ActiveLoop,
    SlotSet,
    ActionExecutionRejected,
    LoopInterrupted,
    FollowupAction,
)
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.policies.rule_policy import RulePolicy, InvalidRule, RULES
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
import pytest
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.shared.core.generator import TrackerWithCachedStates
import rasa.core.policies
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    ActiveLoop,
    SlotSet,
    ActionExecutionRejected,
    LoopInterrupted,
    FollowupAction,
)
from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_BACK_NAME,
    RULE_SNIPPET_ACTION_NAME,
    REQUESTED_SLOT,
    USER,
    PREVIOUS_ACTION,
    ACTIVE_LOOP,
    LOOP_NAME,
    RULE_ONLY_SLOTS,
    RULE_ONLY_LOOPS,
)
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.shared.core.trackers import DialogueStateTracker


def test_memo_with_multiple_slots():
    intent_1 = "intent_1"
    utter_1 = "utter_1"
    utter_2 = "utter_2"
    value_1 = "value_1"
    value_2 = "value_2"
    slot_1 = "entity_1"
    slot_2 = "entity_2"
    entity_1 = "card_1"
    entity_2 = "card_2"
    domain = Domain.from_yaml(
        f"""
            version: "2.0"
            intents:
            - {intent_1}
            actions:
            - {utter_1}
            - {utter_2}
            entities:
            - {entity_1}
            - {entity_2}
            slots:
              {slot_1}:
                type: text
                influence_conversation: true
              {slot_2}:
                type: text
                influence_conversation: true
            """
    )
    story = TrackerWithCachedStates.from_events(
        "story without action_listen",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                intent={"name": intent_1},
                entities=[{"entity": entity_1}, {"entity": entity_2}],
            ),
            # SlotSet(slot_1, value_1),
            # SlotSet(slot_2, value_2),
            ActionExecuted(utter_1),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=False,
    )
    policy = MemoizationPolicy()
    policy.train([story], domain, RegexInterpreter())

    print("================policy trained now pls infer")

    # the order of slots set doesn't matter for prediction
    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(
            "haha",
            intent={"name": intent_1},
            entities=[
                {"entity": entity_2, "value": value_2},
                {"entity": entity_1, "value": value_1},
            ],
        ),
        # SlotSet(slot_1, value_1),
        # SlotSet(slot_2, value_2),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, utter_1)
    assert False


def assert_predicted_action(
    prediction: PolicyPrediction,
    domain: Domain,
    expected_action_name: Text,
    confidence: float = 1.0,
    is_end_to_end_prediction: bool = False,
    is_no_user_prediction: bool = False,
) -> None:
    assert prediction.max_confidence == confidence
    index_of_predicted_action = prediction.max_confidence_index
    prediction_action_name = domain.action_names_or_texts[index_of_predicted_action]
    assert prediction_action_name == expected_action_name
    assert prediction.is_end_to_end_prediction == is_end_to_end_prediction
    assert prediction.is_no_user_prediction == is_no_user_prediction
