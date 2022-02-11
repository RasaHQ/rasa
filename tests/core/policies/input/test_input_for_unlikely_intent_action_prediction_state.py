from typing import Optional

import pytest

from rasa.core.turns.state.state_featurizers import BasicStateFeaturizer

from rasa.core.policies.input.input_for_unlikely_intent_prediction import (
    InputFeaturesCreatorForUnlikelyIntentActionPrediction,
)
from rasa.shared.core.domain import Domain
from tests.core.utilities import user_uttered

from rasa.shared.nlu.constants import INTENT, ACTION_NAME
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
    USER,
    PREVIOUS_ACTION,
)
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.mark.parametrize("max_history", [None, 2])
def test_prediction_states_base_examples(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):

    # IntentMaxHistoryTrackerFeaturizer prediction is only done after
    # a UserUttered event so remove the last BotUttered and
    # ActionExecuted events.
    moodbot_tracker = moodbot_tracker.copy()
    moodbot_tracker.events.pop()
    moodbot_tracker.events.pop()

    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        state_featurizer=BasicStateFeaturizer(),
    )
    actual_states = input_creator._create_inference_states_only(
        moodbot_tracker,
        moodbot_domain,
    )

    expected_states = [
        {},
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "greet"},
        },
        {
            USER: {INTENT: "greet"},
            PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},
        },
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "mood_unhappy"},
        },
        {
            USER: {INTENT: "mood_unhappy"},
            PREVIOUS_ACTION: {ACTION_NAME: "utter_cheer_up"},
        },
        {
            USER: {INTENT: "mood_unhappy"},
            PREVIOUS_ACTION: {ACTION_NAME: "utter_did_that_help"},
        },
    ]
    if max_history is not None:
        expected_states = expected_states[-max_history:]

    assert actual_states is not None
    assert actual_states == expected_states


@pytest.mark.parametrize("max_history", [None, 2])
def test_prediction_states_hide_rule_states(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):
    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        ignore_rule_only_turns=True,
        state_featurizer=BasicStateFeaturizer(),
    )

    rule_tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet"),
            ActionExecuted("utter_greet", hide_rule_turn=True),
            ActionExecuted(ACTION_LISTEN_NAME, hide_rule_turn=True),
        ],
        domain=moodbot_domain,
    )

    actual_states = input_creator._create_inference_states_only(
        rule_tracker,
        moodbot_domain,
    )

    expected_states = [{}]

    assert actual_states is not None
    assert actual_states == expected_states

    embedded_rule_tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet"),
            ActionExecuted("utter_greet", hide_rule_turn=True),
            ActionExecuted(ACTION_LISTEN_NAME, hide_rule_turn=True),
            user_uttered("mood_great"),
        ],
        domain=moodbot_domain,
    )

    actual_states = input_creator._create_inference_states_only(
        embedded_rule_tracker,
        moodbot_domain,
    )

    expected_states = [
        {},
    ]

    assert actual_states is not None
    assert actual_states == expected_states


@pytest.mark.parametrize("max_history", [None, 3])
def test_prediction_states_ignores_action_intent_unlikely(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):
    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        ignore_action_unlikely_intent=True,
        state_featurizer=BasicStateFeaturizer(),
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet"),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ActionExecuted("utter_greet"),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("mood_great"),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ActionExecuted("utter_happy"),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("goodbye"),
        ],
        domain=moodbot_domain,
    )

    actual_states = input_creator._create_inference_states_only(
        tracker,
        moodbot_domain,
    )

    expected_states = [
        {},
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "greet"},
        },
        {
            USER: {INTENT: "greet"},
            PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},
        },
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "mood_great"},
        },
        {
            USER: {INTENT: "mood_great"},
            PREVIOUS_ACTION: {ACTION_NAME: "utter_happy"},
        },
    ]

    if max_history is not None:
        expected_states = expected_states[-max_history:]

    assert actual_states is not None
    assert actual_states == expected_states


@pytest.mark.parametrize("max_history", [None, 3])
def test_prediction_states_keeps_action_intent_unlikely(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):
    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        state_featurizer=BasicStateFeaturizer(),
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet"),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ActionExecuted("utter_greet"),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("mood_great"),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ActionExecuted("utter_happy"),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("goodbye"),
        ],
        domain=moodbot_domain,
    )

    actual_states = input_creator._create_inference_states_only(tracker, moodbot_domain)

    expected_states = [
        {},
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "greet"},
        },
        {
            USER: {INTENT: "greet"},
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_UNLIKELY_INTENT_NAME},
        },
        {
            USER: {INTENT: "greet"},
            PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},
        },
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "mood_great"},
        },
        {
            USER: {INTENT: "mood_great"},
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_UNLIKELY_INTENT_NAME},
        },
        {
            USER: {INTENT: "mood_great"},
            PREVIOUS_ACTION: {ACTION_NAME: "utter_happy"},
        },
    ]

    if max_history is not None:
        expected_states = expected_states[-max_history:]

    assert actual_states is not None
    assert actual_states == expected_states
