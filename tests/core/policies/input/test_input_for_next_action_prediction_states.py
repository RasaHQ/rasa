from typing import Optional

import pytest

from rasa.core.policies.input.input_for_next_action_prediction import (
    InputStatesCreatorForNextActionPrediction,
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


# FIXME: rework these tests
#  - move the repeated checks to a single method
#  - make explicit where the same input data is used and different outputs are
#    expected (cf. unlikely intent)
#  - add tests for training data generation (do not exist currently -- only implicit
#    testing via other rule policy related tests)


@pytest.mark.parametrize("max_history", [None, 2])
def test_prediction_states(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):
    input_creator = InputStatesCreatorForNextActionPrediction(max_history=max_history)
    actual_states = input_creator.create_inference_data(
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
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "deny"},
        },
        {
            USER: {INTENT: "deny"},
            PREVIOUS_ACTION: {ACTION_NAME: "utter_goodbye"},
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

    # FIXME: separate the two test cases in here

    input_creator = InputStatesCreatorForNextActionPrediction(
        max_history=max_history,
        ignore_rule_only_turns=True,
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

    actual_states = input_creator.create_inference_data(
        rule_tracker,
        moodbot_domain,
    )

    expected_states = [
        {},
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "greet"},
        },
    ]

    assert actual_states is not None
    assert len(actual_states) == len(expected_states)

    for actual, expected in zip(actual_states, expected_states):
        assert actual == expected

    embedded_rule_tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet"),
            ActionExecuted("utter_greet", hide_rule_turn=True),
            ActionExecuted(ACTION_LISTEN_NAME, hide_rule_turn=True),
            user_uttered("mood_great"),
            ActionExecuted("utter_happy"),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        domain=moodbot_domain,
    )

    actual_states = input_creator.create_inference_data(
        embedded_rule_tracker,
        moodbot_domain,
    )

    expected_states = [
        {},
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "mood_great"},
        },
        {
            USER: {INTENT: "mood_great"},
            PREVIOUS_ACTION: {ACTION_NAME: "utter_happy"},
        },
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "mood_great"},
        },
    ]

    if max_history is not None:
        expected_states = expected_states[-max_history:]

    assert actual_states is not None
    assert actual_states == expected_states


@pytest.mark.parametrize("max_history", [None, 3])
def test_prediction_states_ignores_action_intent_unlikely(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):

    input_creator = InputStatesCreatorForNextActionPrediction(
        max_history=max_history, ignore_action_unlikely_intent=True
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

    actual_states = input_creator.create_inference_data(
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
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "goodbye"},
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

    input_creator = InputStatesCreatorForNextActionPrediction(max_history=max_history)

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

    actual_states = input_creator.create_inference_data(
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
        {
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
            USER: {INTENT: "goodbye"},
        },
    ]

    if max_history is not None:
        expected_states = expected_states[-max_history:]

    assert actual_states is not None
    assert actual_states == expected_states
