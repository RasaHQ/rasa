from typing import Text
import os

import numpy as np
import pytest

from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.events import ActionExecuted, BotUttered, UserUttered
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    get_events_and_states,
    TrackerStateExtractor,
    TrackerFeaturizer,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import RegexInterpreter
from tests.core.utilities import tracker_from_dialogue_file


# def test_dummy():
#     domain_path = os.path.join("data", "test_moodbot", "domain.yml")
#     moodbot_domain = Domain.load(domain_path)
#     tracker = tracker_from_dialogue_file(
#         "data/test_dialogues/moodbot.json", moodbot_domain
#     )
#     breakpoint()


def test_get_events_and_states_aligns_correctly():
    domain_path = os.path.join("data", "test_moodbot", "domain.yml")
    moodbot_domain = Domain.load(domain_path)

    # Note: A trackers applied_events returns only ActionExecuted and UserUttered
    # events. Similarly, we need to use only those events to create a tracker
    # from DialogueStateTracker.from_events.
    events = [
        ActionExecuted(action_name=ACTION_LISTEN_NAME),
        UserUttered(text="Hi talk to me", intent={INTENT_NAME_KEY: "greet"},),
        ActionExecuted(
            action_text=moodbot_domain.responses["utter_greet"][0]["text"]
        ),  # FIXME: this cannot be right
        ActionExecuted(action_name=ACTION_LISTEN_NAME),
        UserUttered(text="Super sad", intent={INTENT_NAME_KEY: "mood_unhappy"},),
        ActionExecuted(
            action_text=moodbot_domain.responses["utter_cheer_up"][0]["text"]
        ),  # FIXME: this cannot be right
    ]

    # We expect that `_get_events_and_states` computes an alignment of all
    # (non-empty) states with the last event that is "included" in that state.
    expected_events_and_partial_states = [
        (
            events[1],
            {
                "user": {"intent": "greet"},
                "prev_action": {"action_name": ACTION_LISTEN_NAME},
            },
        ),
        (
            events[2],
            {
                "user": {"intent": "greet"},
                "prev_action": {"action_text": events[2].action_text},
            },
        ),
        (
            events[4],
            {
                "user": {"intent": "mood_unhappy"},
                "prev_action": {"action_name": ACTION_LISTEN_NAME},
            },
        ),
        (
            events[5],
            {
                "user": {"intent": "mood_unhappy"},
                "prev_action": {"action_text": events[5].action_text},
            },
        ),
    ]

    tracker = DialogueStateTracker.from_events(sender_id="dummy_sender", evts=events)
    _, _, alignment = get_events_and_states(
        tracker, domain=moodbot_domain, omit_unset_slots=False
    )

    assert len(alignment) == len(expected_events_and_partial_states)

    for idx in range(len(alignment)):
        msg = f"at idx {idx}"
        expected_event, expected_partial_state = expected_events_and_partial_states[idx]
        state, event = alignment[idx]  # Note: state is first here
        assert event == expected_event, msg
        for key, expected_partial_state in expected_partial_state.items():
            assert key in state, msg
            for attribute in expected_partial_state:
                assert attribute in state[key], msg
                assert state[key][attribute] == expected_partial_state[attribute], msg


"""
def test_fail_to_load_non_existent_featurizer():
    assert TrackerFeaturizer.load("non_existent_class") is None


def test_persist_and_load_tracker_featurizer(tmp_path: Text, moodbot_domain: Domain):
    state_featurizer = SingleStateFeaturizer()
    state_featurizer.prepare_for_training(moodbot_domain, RegexInterpreter())
    tracker_featurizer = TrackerFeaturizer(state_featurizer)

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
        tracker_featurizer.featurize_trackers_for_training(
            [], domain, RegexInterpreter()
        )


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
    tracker_featurizer = TrackerFeaturizer(state_featurizer)

    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )
    (
        state_features,
        labels,
        entity_tags,
    ) = tracker_featurizer.featurize_trackers_for_training(
        [tracker], moodbot_domain, RegexInterpreter()
    )

    assert state_features is not None
    assert len(state_features) == 7
    assert labels is not None
    assert len(labels) == 7
    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])
"""
