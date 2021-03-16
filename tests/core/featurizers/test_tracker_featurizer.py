from typing import Text

import numpy as np
import pytest

from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import RegexInterpreter
from tests.core.utilities import tracker_from_dialogue_file


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


def test_convert_labels_to_ids(domain: Domain):
    trackers_as_actions = [
        ["utter_greet", "utter_channel"],
        ["utter_greet", "utter_default", "utter_goodbye"],
    ]

    tracker_featurizer = TrackerFeaturizer()

    actual_output = tracker_featurizer._convert_labels_to_ids(
        trackers_as_actions, domain
    )
    expected_output = np.array([np.array([14, 11]), np.array([14, 12, 13])])

    assert expected_output.size == actual_output.size
    for expected_array, actual_array in zip(expected_output, actual_output):
        assert np.all(expected_array == actual_array)


def test_featurize_trackers_raises_on_missing_state_featurizer(domain: Domain):
    tracker_featurizer = TrackerFeaturizer()

    with pytest.raises(ValueError):
        tracker_featurizer.featurize_trackers([], domain, RegexInterpreter())


def test_featurize_trackers_with_full_dialogue_tracker_featurizer(
    moodbot_domain: Domain,
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = FullDialogueTrackerFeaturizer(state_featurizer)

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )
    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker], moodbot_domain, RegexInterpreter()
    )

    assert state_features is not None
    assert len(state_features) == 1
    assert labels is not None
    assert len(labels) == 1
    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


def test_featurize_trackers_with_max_history_tracker_featurizer(moodbot_domain: Domain):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(state_featurizer)

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )
    state_features, labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker], moodbot_domain, RegexInterpreter()
    )

    assert state_features is not None
    assert len(state_features) == 7
    assert labels is not None
    assert len(labels) == 7
    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])
