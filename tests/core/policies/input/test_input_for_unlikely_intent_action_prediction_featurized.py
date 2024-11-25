from typing import Text, Dict, List, Optional

import numpy as np
import pytest

from rasa.core.turns.state.state_featurizers import BasicStateFeaturizer
from rasa.core.policies.input.input_for_unlikely_intent_prediction import (
    InputFeaturesCreatorForUnlikelyIntentActionPrediction,
)
from tests.core.policies.input.utils import compare_featurized_states
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


@pytest.mark.parametrize(
    "max_history,moodbot_features",
    [
        [None, "IntentTokenizerSingleStateFeaturizer"],
        [2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
def test_create_training_data_basic(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):
    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        state_featurizer=BasicStateFeaturizer(),
    )

    input_creator.train(domain=moodbot_domain)

    actual_features, actual_labels, entity_tags = input_creator.create_training_data(
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
            {ACTION_NAME: [moodbot_features["actions"]["utter_cheer_up"]]},
            {ACTION_NAME: [moodbot_features["actions"]["utter_did_that_help"]]},
        ],
    ]
    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]

    assert actual_features is not None
    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)

    expected_labels = np.array([[5, 7, 3]]).T

    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape
    assert np.all(actual_labels == expected_labels)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize(
    "max_history, moodbot_features",
    [
        [None, "IntentTokenizerSingleStateFeaturizer"],
        [2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
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

    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        state_featurizer=BasicStateFeaturizer(),
        ignore_action_unlikely_intent=True,
    )

    input_creator.train(domain=moodbot_domain)

    actual_features, actual_labels, entity_tags = input_creator.create_training_data(
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
            {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
        ],
    ]

    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]

    assert actual_features is not None
    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)

    expected_labels = np.array([[5, 7]]).T
    assert actual_labels.shape == expected_labels.shape
    for actual, expected in zip(actual_labels, expected_labels):
        assert np.all(actual == expected)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize(
    "max_history,moodbot_features",
    [
        [None, "IntentTokenizerSingleStateFeaturizer"],
        [2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
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
    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        state_featurizer=BasicStateFeaturizer(),
    )

    input_creator.train(domain=moodbot_domain)

    actual_features, actual_labels, entity_tags = input_creator.create_training_data(
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

    expected_labels = np.array([[5, 7]]).T
    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape

    for actual, expected in zip(actual_labels, expected_labels):
        assert np.all(actual == expected)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize(
    "remove_duplicates,max_history,moodbot_features",
    [
        [True, None, "IntentTokenizerSingleStateFeaturizer"],
        [True, 2, "IntentTokenizerSingleStateFeaturizer"],
        [False, None, "IntentTokenizerSingleStateFeaturizer"],
        [False, 2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
def test_create_training_data_deduplicate(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    remove_duplicates: bool,
    max_history: Optional[int],
):
    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        ignore_duplicate_input_turns=remove_duplicates,
        state_featurizer=BasicStateFeaturizer(),
    )

    input_creator.train(domain=moodbot_domain)

    # Add Duplicate moodbot_tracker states should get removed.
    actual_features, actual_labels, entity_tags = input_creator.create_training_data(
        [moodbot_tracker, moodbot_tracker],
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
            {ACTION_NAME: [moodbot_features["actions"]["utter_cheer_up"]]},
            {ACTION_NAME: [moodbot_features["actions"]["utter_did_that_help"]]},
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

    expected_labels = np.array([[5, 7, 3]]).T
    if not remove_duplicates:
        expected_labels = np.vstack([expected_labels] * 2)

    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape
    assert np.all(actual_labels == expected_labels)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize(
    "remove_duplicates, max_history",
    [
        [True, None],
        [True, 2],
        [False, None],
        [False, 2],
    ],
)
def test_create_training_with_multilabels(
    moodbot_domain: Domain, max_history: Optional[int], remove_duplicates: bool
):
    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        ignore_duplicate_input_turns=remove_duplicates,
        state_featurizer=BasicStateFeaturizer(),
    )

    input_creator.train(domain=moodbot_domain)

    event_list1 = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet"),
        ActionExecuted("utter_greet"),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("mood_great"),
    ]
    tracker1 = DialogueStateTracker.from_events(
        "default", event_list1, domain=moodbot_domain
    )
    event_list2 = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet"),
        ActionExecuted("utter_greet"),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("mood_unhappy"),
    ]
    tracker2 = DialogueStateTracker.from_events(
        "default", event_list2, domain=moodbot_domain
    )

    _1, actual_labels, _2 = input_creator.create_training_data(
        [tracker1, tracker2],
        moodbot_domain,
        precomputations=None,
    )

    greet_index = 5
    mood_great_index = 6
    mood_unhappy_index = 7

    if remove_duplicates:
        expected_labels = np.array(
            [
                [greet_index, -1],
                [mood_great_index, mood_unhappy_index],
            ]
        )
    else:
        expected_labels = np.array(
            [
                [greet_index, -1],
                [mood_great_index, mood_unhappy_index],
                [greet_index, -1],
                [mood_great_index, mood_unhappy_index],
            ]
        )

    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape

    # Order of label indices may be different,
    # hence need to sort the indices and then check.
    for actual_label_indices, expected_label_indices in zip(
        actual_labels, expected_labels
    ):
        assert sorted(actual_label_indices) == sorted(expected_label_indices)


@pytest.mark.parametrize(
    "max_history,moodbot_features",
    [
        [None, "IntentTokenizerSingleStateFeaturizer"],
        [2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
def test_create_inference_data_basic(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
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

    input_creator.train(domain=moodbot_domain)

    actual_features = input_creator.create_inference_data(
        moodbot_tracker,
        moodbot_domain,
        precomputations=None,
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
    ]
    if max_history is not None:
        expected_features = expected_features[-max_history:]

    assert actual_features is not None
    compare_featurized_states(actual_features, expected_features)


@pytest.mark.parametrize(
    "max_history,moodbot_features",
    [
        [None, "IntentTokenizerSingleStateFeaturizer"],
        [2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
def test_create_inference_data_ignore_action_unlikely(
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
    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        state_featurizer=BasicStateFeaturizer(),
        ignore_action_unlikely_intent=True,
    )

    input_creator.train(domain=moodbot_domain)

    actual_features = input_creator.create_inference_data(
        tracker,
        moodbot_domain,
        precomputations=None,
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
    ]
    if max_history is not None:
        expected_features = expected_features[-max_history:]

    assert actual_features is not None
    assert compare_featurized_states(actual_features, expected_features)


@pytest.mark.parametrize(
    "max_history,moodbot_features",
    [
        [None, "IntentTokenizerSingleStateFeaturizer"],
        [2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
def test_create_inference_data_keep_action_unlikely(
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
    input_creator = InputFeaturesCreatorForUnlikelyIntentActionPrediction(
        max_history=max_history,
        state_featurizer=BasicStateFeaturizer(),
    )

    input_creator.train(domain=moodbot_domain)
    actual_features = input_creator.create_inference_data(
        tracker,
        moodbot_domain,
        precomputations=None,
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
    ]

    if max_history is not None:
        expected_features = expected_features[-max_history:]

    assert actual_features is not None
    assert compare_featurized_states(actual_features, expected_features)
