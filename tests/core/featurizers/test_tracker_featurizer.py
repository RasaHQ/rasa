from typing import Text

import numpy as np
from scipy import sparse
import pytest

from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
    IntentMaxHistoryTrackerFeaturizer
)
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import RegexInterpreter
from tests.core.utilities import tracker_from_dialogue_file, user_uttered
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import INTENT, ACTION_NAME, FEATURE_TYPE_SENTENCE
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import DialogueStateTracker


def test_fail_to_load_non_existent_featurizer():
    assert TrackerFeaturizer.load("non_existent_class") is None


def test_persist_and_load_tracker_featurizer(tmp_path: Text, moodbot_domain: Domain):
    state_featurizer = SingleStateFeaturizer()
    state_featurizer.prepare_for_training(moodbot_domain, RegexInterpreter())
    tracker_featurizer = MaxHistoryTrackerFeaturizer(state_featurizer)

    tracker_featurizer.persist(tmp_path)

    loaded_tracker_featurizer = TrackerFeaturizer.load(tmp_path)

    assert loaded_tracker_featurizer is not None
    assert loaded_tracker_featurizer.state_featurizer is not None


def test_convert_action_labels_to_ids(domain: Domain):
    trackers_as_actions = [
        ["utter_greet", "utter_channel"],
        ["utter_greet", "utter_default", "utter_goodbye"],
    ]

    tracker_featurizer = TrackerFeaturizer()

    actual_output = tracker_featurizer._convert_labels_to_ids(
        trackers_as_actions, domain
    )
    expected_output = np.array([np.array([15, 12]), np.array([15, 13, 14])])

    assert expected_output.size == actual_output.size
    for expected_array, actual_array in zip(expected_output, actual_output):
        assert np.all(expected_array == actual_array)


def test_featurize_trackers_raises_on_missing_state_featurizer(domain: Domain):
    tracker_featurizer = TrackerFeaturizer()

    with pytest.raises(ValueError):
        tracker_featurizer.featurize_trackers([], domain, RegexInterpreter())


@pytest.fixture
def make_moodbot_example(moodbot_domain):
    intent_shape = (1, len(moodbot_domain.intents))
    action_shape = (1, len(moodbot_domain.action_names_or_texts))
    action_listen = Features(
        sparse.coo_matrix(([1.0], [[0], [0]]), shape=action_shape),
        FEATURE_TYPE_SENTENCE, 
        ACTION_NAME, 
        "SingleStateFeaturizer"
    )
    utter_greet = Features(
        sparse.coo_matrix(([1.0], [[0], [15]]), shape=action_shape),
        FEATURE_TYPE_SENTENCE, 
        ACTION_NAME, 
        "SingleStateFeaturizer"
    )
    utter_cheer_up = Features(
        sparse.coo_matrix(([1.0], [[0], [12]]), shape=action_shape),
        FEATURE_TYPE_SENTENCE, 
        ACTION_NAME, 
        "SingleStateFeaturizer"
    )
    utter_did_that_help = Features(
        sparse.coo_matrix(([1.0], [[0], [13]]), shape=action_shape),
        FEATURE_TYPE_SENTENCE, 
        ACTION_NAME, 
        "SingleStateFeaturizer"
    )
    utter_goodbye = Features(
        sparse.coo_matrix(([1.0], [[0], [14]]), shape=action_shape),
        FEATURE_TYPE_SENTENCE, 
        ACTION_NAME, 
        "SingleStateFeaturizer"
    )
    greet = Features(
        sparse.coo_matrix(([1.0], [[0], [5]]), shape=intent_shape),
        FEATURE_TYPE_SENTENCE, 
        INTENT, 
        "SingleStateFeaturizer"
    )
    mood_unhappy = Features(
        sparse.coo_matrix(([1.0], [[0], [7]]), shape=intent_shape),
        FEATURE_TYPE_SENTENCE, 
        INTENT, 
        "SingleStateFeaturizer"
    )
    deny = Features(
        sparse.coo_matrix(([1.0], [[0], [3]]), shape=intent_shape),
        FEATURE_TYPE_SENTENCE, 
        INTENT, 
        "SingleStateFeaturizer"
    )
 
    action_labels = np.array([0, 15, 0, 12, 13, 0, 14]).reshape(7,1)
    intent_labels = np.array([5, 7, 3]).reshape(3,1)
    entities = [{} for _ in range(7)]
    features = [
            {}, 
            {'intent': [greet], 'action_name': [action_listen]}, 
            {'action_name': [utter_greet]}, 
            {"intent": [mood_unhappy], "action_name": [action_listen]}, 
            {"action_name": [utter_cheer_up]}, 
            {"action_name": [utter_did_that_help]}, 
            {"intent": [deny], "action_name": [action_listen]},
            {"action_name": [utter_goodbye]}, 
    ]   
    states = [
        {},
        {
            'user': {'intent': 'greet'}, 
            'prev_action': {'action_name': 'action_listen'}
        },
        {
            'user': {'intent': 'greet'}, 
            'prev_action': {'action_name': 'utter_greet'},
        },
        {
            'user': {'intent': 'mood_unhappy'}, 
            'prev_action': {'action_name': 'action_listen'}
        },
        {
            'user': {'intent': 'mood_unhappy'}, 
            'prev_action': {'action_name': 'utter_cheer_up'}
        },
        {
            'user': {'intent': 'mood_unhappy'}, 
            'prev_action': {'action_name': 'utter_did_that_help'}
        },
        {
            'user': {'intent': 'deny'}, 
            'prev_action': {'action_name': 'action_listen'}
        },
        {
            'user': {'intent': 'deny'}, 
            'prev_action': {'action_name': 'utter_goodbye'}
        }
    ]
    def _make_moodbot_example(
        max_history=-1, 
        label_type="action", 
        duplicate=False, 
        remove_final_action=True,
        prediction_features=False,
    ):

        if remove_final_action:
            example_features = features[:-1]
            example_states = states[:-1]
        else:
            example_features = features
            example_states = states

  
        example = {"entities": entities}
        if max_history == -1:
            example["features"] = [example_features]
            example["states"] = [example_states]
        elif max_history == None:
            example["features"] = [
                example_features[:i + 1]
                for i in range(len(example_features))
            ]
            example["states"] = [
                example_states[:i + 1]
                for i in range(len(example_states))
            ]      

        else:
            example["features"] = [
                example_features[max(0, i - max_history + 1):i + 1]
                for i in range(len(example_features))
            ]
            example["states"] = [
                example_states[max(0, i - max_history + 1):i + 1]
                for i in range(len(example_states))
            ]

        if label_type == "intent":
            example["features"] = [
                x for i, x in enumerate(example["features"][:-1])
                if "intent" in example["features"][i+1][-1]
            ]
            example["states"] = [
                states for i, states in enumerate(example["states"][:-1])
                if example["states"][i+1][-1]["prev_action"] == {"action_name": "action_listen"}
            ]
#                example_features[:i + 1]
#                for i in range(len(example_features))
#            ]
 
        if prediction_features:
            example["features"] = [example["features"][-1]]
            example["states"] = [example["states"][-1]]
        if label_type == "action":
            example["labels"] = action_labels
        else: 
            example["labels"] = intent_labels
        if max_history == -1:
            example["labels"] = example["labels"].reshape(1, -1)

        if duplicate:
            example["entities"] = example["entities"] + example["entities"]
            example["features"] = example["features"] + example["features"]
            example["states"] = example["states"] + example["states"]
            example["labels"] = np.vstack([example["labels"], example["labels"]])

        return example

    return _make_moodbot_example


def compare_featurized_states(states1, states2):
    if len(states1) != len(states2):
        return False

    for state1, state2 in zip(states1, states2):
        if state1.keys() != state2.keys():
            return False
        for key in state1.keys():
            for feature1, feature2 in zip(state1[key], state2[key]):
                if np.any((feature1.features != feature2.features).toarray()):
                    return False
                if feature1.origin != feature2.origin:
                    return False
                if feature1.attribute != feature2.attribute:
                    return False
                if feature1.type != feature2.type:
                    return False

    return True


def test_featurize_trackers_with_full_dialogue_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = FullDialogueTrackerFeaturizer(state_featurizer)

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )
    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker], moodbot_domain, RegexInterpreter()
    )

    expected_example = make_moodbot_example()

    assert state_features is not None
    assert len(state_features) == 1
    
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )
    assert labels is not None
    assert len(labels) == 1
    for actual_label, expected_label in zip(labels, expected_example["labels"]):
        assert np.all(actual_label == expected_label)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


def test_create_state_features_with_full_dialogue_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = FullDialogueTrackerFeaturizer(state_featurizer)

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    interpreter = RegexInterpreter()
    state_featurizer.prepare_for_training(moodbot_domain, interpreter)
    state_features = tracker_featurizer.create_state_features(
        [tracker], moodbot_domain, interpreter
    )

    expected_example = make_moodbot_example(remove_final_action=False)

    assert state_features is not None
    assert len(state_features) == len(expected_example["features"])
    
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )


def test_prediction_states_with_full_dialogue_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = FullDialogueTrackerFeaturizer(state_featurizer)

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    states = tracker_featurizer.prediction_states(
        [tracker], moodbot_domain,
    )

    expected_example = make_moodbot_example(
        remove_final_action=False,
        prediction_features=True
    )

    assert states is not None
    assert len(states) == len(expected_example["states"])
   
    for actual_states, expected_states in zip(
        states, 
        expected_example["states"]
    ):
        assert actual_states == expected_states


@pytest.mark.parametrize("max_history", [None, 2])
def test_featurize_trackers_with_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker], moodbot_domain, RegexInterpreter()
    )
    
    expected_example = make_moodbot_example(max_history=max_history)

    assert state_features is not None
    assert len(state_features) == len(expected_example["features"])
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )

    assert labels is not None
    assert labels.shape == expected_example["labels"].shape
    assert np.all(labels == expected_example["labels"])

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_deduplicate_featurize_trackers_with_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history,
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    # add duplicate tracker
    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker, tracker], moodbot_domain, RegexInterpreter()
    )
    
    expected_example = make_moodbot_example(max_history=max_history)

    assert state_features is not None
    assert len(state_features) == len(expected_example["features"])
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )

    assert labels is not None
    assert labels.shape == expected_example["labels"].shape
    assert np.all(labels == expected_example["labels"])

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_no_deduplicate_featurize_trackers_with_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history,
        remove_duplicates=False
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    # add duplicate tracker
    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker, tracker], moodbot_domain, RegexInterpreter()
    )
    
    expected_example = make_moodbot_example(max_history=max_history, duplicate=True)

    assert state_features is not None
    assert len(state_features) == len(expected_example["features"])
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )

    assert labels is not None
    assert labels.shape == expected_example["labels"].shape
    assert np.all(labels == expected_example["labels"])

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_state_features_with_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    interpreter = RegexInterpreter()
    state_featurizer.prepare_for_training(moodbot_domain, interpreter)
    state_features = tracker_featurizer.create_state_features(
        [tracker], moodbot_domain, interpreter
    )

    expected_example = make_moodbot_example(
        max_history=max_history,
        remove_final_action=False,
        prediction_features=True
    )

    assert state_features is not None
    assert len(state_features) == len(expected_example["features"])
    
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )


@pytest.mark.parametrize("max_history", [None, 2])
def test_prediction_states_with_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    states = tracker_featurizer.prediction_states(
        [tracker], moodbot_domain,
    )

    expected_example = make_moodbot_example(
        max_history=max_history,
        remove_final_action=False,
        prediction_features=True
    )

    assert states is not None
    assert len(states) == len(expected_example["states"])
   
    for actual_states, expected_states in zip(
        states, 
        expected_example["states"]
    ):
        assert actual_states == expected_states


@pytest.mark.parametrize("max_history", [None, 2])
def test_featurize_trackers_with_intent_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker], moodbot_domain, RegexInterpreter()
    )
    
    expected_example = make_moodbot_example(
        max_history=max_history, 
        label_type="intent"
    )

    assert state_features is not None
    assert len(state_features) == len(expected_example["features"])
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )

    assert labels is not None
    assert labels.shape == expected_example["labels"].shape
    assert np.all(labels == expected_example["labels"])

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_deduplicate_featurize_trackers_with_intent_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history,
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    # add duplicate tracker
    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker, tracker], moodbot_domain, RegexInterpreter()
    )
    
    expected_example = make_moodbot_example(
        max_history=max_history,
        label_type="intent",
    )

    assert state_features is not None
    assert len(state_features) == len(expected_example["features"])
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )

    assert labels is not None
    assert labels.shape == expected_example["labels"].shape
    assert np.all(labels == expected_example["labels"])

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_no_deduplicate_featurize_trackers_with_intent_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history,
        remove_duplicates=False
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    # add duplicate tracker
    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker, tracker], moodbot_domain, RegexInterpreter()
    )
    
    expected_example = make_moodbot_example(
        max_history=max_history, 
        label_type="intent",
        duplicate=True
    )

    assert state_features is not None
    assert len(state_features) == len(expected_example["features"])
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )

    assert labels is not None
    assert labels.shape == expected_example["labels"].shape
    assert np.all(labels == expected_example["labels"])

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_state_features_with_intent_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    interpreter = RegexInterpreter()
    state_featurizer.prepare_for_training(moodbot_domain, interpreter)
    state_features = tracker_featurizer.create_state_features(
        [tracker], moodbot_domain, interpreter
    )

    expected_example = make_moodbot_example(
        max_history=max_history,
        remove_final_action=False,
        label_type="intent",
        prediction_features=True
    )

    assert state_features is not None
    assert len(state_features) == len(expected_example["features"])
    
    for actual_features, expected_features in zip(
        state_features, 
        expected_example["features"]
    ):
        assert compare_featurized_states(
            actual_features, 
            expected_features
        )


@pytest.mark.parametrize("max_history", [None, 2])
def test_prediction_states_with_intent_max_history_tracker_featurizer(
    moodbot_domain: Domain, make_moodbot_example, max_history
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, 
        max_history=max_history
    )

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )

    states = tracker_featurizer.prediction_states(
        [tracker], moodbot_domain,
    )

    expected_example = make_moodbot_example(
        max_history=max_history,
        remove_final_action=False,
        label_type="intent",
        prediction_features=True
    )

    assert states is not None
    assert len(states) == len(expected_example["states"])
   
    for actual_states, expected_states in zip(
        states, 
        expected_example["states"]
    ):
        assert actual_states == expected_states


def test_multilabels_with_intent_max_history_tracker_featurizer(moodbot_domain: Domain):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(state_featurizer)

    event_list1 = [
        ActionExecuted("action_listen"),
        user_uttered("greet"),
        ActionExecuted("utter_greet"),
        ActionExecuted("action_listen"),
        user_uttered("mood_great"),
    ]
    tracker1 = DialogueStateTracker.from_events(
        "default",
        event_list1,
        domain=moodbot_domain
    )
    event_list2 = [
        ActionExecuted("action_listen"),
        user_uttered("greet"),
        ActionExecuted("utter_greet"),
        ActionExecuted("action_listen"),
        user_uttered("mood_unhappy"),
    ]
    tracker2 = DialogueStateTracker.from_events(
        "default",
        event_list2,
        domain=moodbot_domain
    )

    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker1, tracker2], moodbot_domain, RegexInterpreter()
    )

    greet_index = 5
    mood_great_index = 6
    mood_unhappy_index = 7

    expected_labels = np.array([
        [greet_index, -1],
        [mood_great_index, mood_unhappy_index],
        [mood_unhappy_index, mood_great_index],
    ])

    assert np.all(labels == expected_labels)


def test_multilabels_with_intent_max_history_tracker_featurizer_no_dedupe(moodbot_domain: Domain):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer,
        remove_duplicates=False,
    )

    event_list1 = [
        ActionExecuted("action_listen"),
        user_uttered("greet"),
        ActionExecuted("utter_greet"),
        ActionExecuted("action_listen"),
        user_uttered("mood_great"),
    ]
    tracker1 = DialogueStateTracker.from_events(
        "default",
        event_list1,
        domain=moodbot_domain
    )
    event_list2 = [
        ActionExecuted("action_listen"),
        user_uttered("greet"),
        ActionExecuted("utter_greet"),
        ActionExecuted("action_listen"),
        user_uttered("mood_unhappy"),
    ]
    tracker2 = DialogueStateTracker.from_events(
        "default",
        event_list2,
        domain=moodbot_domain
    )

    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker1, tracker2], moodbot_domain, RegexInterpreter()
    )

    greet_index = 5
    mood_great_index = 6
    mood_unhappy_index = 7

    expected_labels = np.array([
        [greet_index, -1],
        [mood_great_index, mood_unhappy_index],
        [greet_index, -1],
        [mood_unhappy_index, mood_great_index],
    ])

    assert np.all(labels == expected_labels)

