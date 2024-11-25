from typing import Text, Dict, List, Optional

import numpy as np
import pytest

from tests.core.policies.input.utils import compare_featurized_states
from rasa.core.turns.state.state_featurizers import BasicStateFeaturizer
from rasa.core.policies.input.input_for_next_action_prediction import (
    InputFeaturesCreatorForNextActionPrediction,
)
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import RegexInterpreter
from tests.core.utilities import user_uttered
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import INTENT, ACTION_NAME
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
)
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_training_data_for_states_with_action_and_intent_only(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):

    input_creator = InputFeaturesCreatorForNextActionPrediction(
        state_featurizer=BasicStateFeaturizer(), max_history=max_history
    )

    input_creator.train(domain=moodbot_domain)

    (actual_features, actual_labels, entity_tags,) = input_creator.create_training_data(
        [moodbot_tracker],
        moodbot_domain,
        precomputations=None,
    )

    expected_features = [
        [
            {},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["mood_unhappy"]],
            },
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["mood_unhappy"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_cheer_up"]]},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["mood_unhappy"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_cheer_up"]]},
            {ACTION_NAME: [moodbot_features["actions"]["utter_did_that_help"]]},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["mood_unhappy"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_cheer_up"]]},
            {ACTION_NAME: [moodbot_features["actions"]["utter_did_that_help"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["deny"]],
            },
        ],
    ]
    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]

    assert actual_features is not None
    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)

    expected_labels = np.array([[0, 16, 0, 13, 14, 0, 15]]).T

    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape
    assert np.all(actual_labels == expected_labels)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_training_data_ignore_action_unlikely_intent(
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):
    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet"),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ActionExecuted("utter_greet"),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("mood_unhappy"),
        ],
        domain=moodbot_domain,
    )
    input_creator = InputFeaturesCreatorForNextActionPrediction(
        state_featurizer=BasicStateFeaturizer(),
        max_history=max_history,
        ignore_action_unlikely_intent=True,
    )
    input_creator.train(domain=moodbot_domain)

    (actual_features, actual_labels, entity_tags,) = input_creator.create_training_data(
        [tracker],
        moodbot_domain,
        precomputations=None,
    )

    expected_features = [
        [
            {},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
        ],
    ]
    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]

    assert actual_features is not None

    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)

    expected_labels = np.array([[0, 16, 0]]).T
    assert actual_labels.shape == expected_labels.shape
    for actual, expected in zip(actual_labels, expected_labels):
        assert np.all(actual == expected)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_training_data_keep_action_unlikely_intent(
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):
    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet"),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ActionExecuted("utter_greet"),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("mood_unhappy"),
        ],
        domain=moodbot_domain,
    )
    input_creator = InputFeaturesCreatorForNextActionPrediction(
        state_featurizer=BasicStateFeaturizer(),
        max_history=max_history,
    )

    input_creator.train(domain=moodbot_domain)

    (actual_features, actual_labels, entity_tags,) = input_creator.create_training_data(
        [tracker],
        moodbot_domain,
        precomputations=None,
    )

    expected_features = [
        [
            {},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"][ACTION_UNLIKELY_INTENT_NAME]]},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"][ACTION_UNLIKELY_INTENT_NAME]]},
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
        ],
    ]
    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]

    assert actual_features is not None
    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)

    expected_labels = np.array([[0, 9, 16, 0]]).T
    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape
    for actual, expected in zip(actual_labels, expected_labels):
        assert np.all(actual == expected)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize(
    "remove_duplicates,max_history",
    [
        [True, None],
        [True, 2],
        [False, None],
        [False, 2],
    ],
)
def test_create_training_data_deduplication(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    remove_duplicates: bool,
    max_history: Optional[int],
):
    input_creator = InputFeaturesCreatorForNextActionPrediction(
        state_featurizer=BasicStateFeaturizer(),
        max_history=max_history,
        ignore_duplicate_turn_label_pairs=remove_duplicates,
    )

    input_creator.train(domain=moodbot_domain)

    # Add Duplicate moodbot_tracker states should get removed.
    (actual_features, actual_labels, entity_tags,) = input_creator.create_training_data(
        [moodbot_tracker, moodbot_tracker], moodbot_domain, precomputations=None
    )

    expected_features = [
        [
            {},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["mood_unhappy"]],
            },
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["mood_unhappy"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_cheer_up"]]},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["mood_unhappy"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_cheer_up"]]},
            {ACTION_NAME: [moodbot_features["actions"]["utter_did_that_help"]]},
        ],
        [
            {},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["greet"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["mood_unhappy"]],
            },
            {ACTION_NAME: [moodbot_features["actions"]["utter_cheer_up"]]},
            {ACTION_NAME: [moodbot_features["actions"]["utter_did_that_help"]]},
            {
                ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
                INTENT: [moodbot_features["intents"]["deny"]],
            },
        ],
    ]
    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]
    if not remove_duplicates:
        expected_features = expected_features * 2

    assert actual_features is not None
    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)

    expected_labels = np.array([[0, 16, 0, 13, 14, 0, 15]]).T
    if not remove_duplicates:
        expected_labels = np.vstack([expected_labels] * 2)

    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape
    assert np.all(actual_labels == expected_labels)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_inference_data_without_action_unlikely_events(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):
    input_creator = InputFeaturesCreatorForNextActionPrediction(
        state_featurizer=BasicStateFeaturizer(), max_history=max_history
    )
    input_creator.train(moodbot_domain)

    actual_features = input_creator.create_inference_data(
        moodbot_tracker,
        moodbot_domain,
        precomputations=None,
        use_text_for_last_user_input=False,
    )

    expected_features = [
        {},
        {
            ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
            INTENT: [moodbot_features["intents"]["greet"]],
        },
        {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
        {
            ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
            INTENT: [moodbot_features["intents"]["mood_unhappy"]],
        },
        {ACTION_NAME: [moodbot_features["actions"]["utter_cheer_up"]]},
        {ACTION_NAME: [moodbot_features["actions"]["utter_did_that_help"]]},
        {
            ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
            INTENT: [moodbot_features["intents"]["deny"]],
        },
        {ACTION_NAME: [moodbot_features["actions"]["utter_goodbye"]]},
    ]
    if max_history is not None:
        expected_features = expected_features[-max_history:]

    assert actual_features is not None
    assert compare_featurized_states(actual_features, expected_features)


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_inference_data_ignore_action_unlikely_intent(
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):
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

    input_creator = InputFeaturesCreatorForNextActionPrediction(
        state_featurizer=BasicStateFeaturizer(),
        max_history=max_history,
        ignore_action_unlikely_intent=True,
    )
    input_creator.train(moodbot_domain)
    actual_features = input_creator.create_inference_data(
        tracker,
        moodbot_domain,
        precomputations=None,
        use_text_for_last_user_input=False,
    )

    expected_features = [
        {},
        {
            ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
            INTENT: [moodbot_features["intents"]["greet"]],
        },
        {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
        {
            ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
            INTENT: [moodbot_features["intents"]["mood_great"]],
        },
        {ACTION_NAME: [moodbot_features["actions"]["utter_happy"]]},
        {
            ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
            INTENT: [moodbot_features["intents"]["goodbye"]],
        },
    ]
    if max_history is not None:
        expected_features = expected_features[-max_history:]

    assert actual_features is not None
    assert compare_featurized_states(actual_features, expected_features)


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_inference_data_keep_action_unlikely_intent(
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):
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

    input_creator = InputFeaturesCreatorForNextActionPrediction(
        state_featurizer=BasicStateFeaturizer(), max_history=max_history
    )

    input_creator.train(domain=moodbot_domain)

    actual_features = input_creator.create_inference_data(
        tracker,
        moodbot_domain,
        precomputations=None,
        use_text_for_last_user_input=False,
    )

    expected_features = [
        {},
        {
            ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
            INTENT: [moodbot_features["intents"]["greet"]],
        },
        {ACTION_NAME: [moodbot_features["actions"][ACTION_UNLIKELY_INTENT_NAME]]},
        {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
        {
            ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
            INTENT: [moodbot_features["intents"]["mood_great"]],
        },
        {ACTION_NAME: [moodbot_features["actions"][ACTION_UNLIKELY_INTENT_NAME]]},
        {ACTION_NAME: [moodbot_features["actions"]["utter_happy"]]},
        {
            ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
            INTENT: [moodbot_features["intents"]["goodbye"]],
        },
    ]
    if max_history is not None:
        expected_features = expected_features[-max_history:]

    assert actual_features is not None
    assert compare_featurized_states(actual_features, expected_features)
