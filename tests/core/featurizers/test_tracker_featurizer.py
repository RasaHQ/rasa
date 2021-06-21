from typing import Text, Dict, List, Optional, Any

import numpy as np
from scipy import sparse
import pytest

from rasa.core.featurizers.single_state_featurizer import (
    SingleStateFeaturizer,
    IntentTokenizerSingleStateFeaturizer,
)
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
    IntentMaxHistoryTrackerFeaturizer,
)
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import RegexInterpreter
from tests.core.utilities import user_uttered
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import INTENT, ACTION_NAME, FEATURE_TYPE_SENTENCE
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
    USER,
    PREVIOUS_ACTION,
)
from rasa.shared.core.events import ActionExecuted, Event
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.tensorflow.constants import LABEL_PAD_ID

import tests.core.utilities


def compare_featurized_states(
    states1: List[Dict[Text, List[Features]]],
    states2: List[Dict[Text, List[Features]]],
) -> bool:
    """Compares two lists of featurized states.

    Args:
        states1: Featurized states
        states2: More featurized states

    Returns:
        True if states are identical and False otherwise.
    """

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


def count_features(
    features: List[List[Dict[Text, List[Features]]]],
    target_feature: Dict[Text, List[Features]],
) -> int:
    num_occurences = 0
    for tracker_features in features:
        for feature in tracker_features:
            if compare_featurized_states([feature], [target_feature]):
                num_occurences += 1
    return num_occurences


def compare_features_and_labels(
    actual_features: Optional[List[List[Dict[Text, List[Features]]]]],
    expected_features: Optional[List[List[Dict[Text, List[Features]]]]],
    actual_labels: np.array,
    expected_labels: np.array,
    actual_entity_tags: Optional[List[List[Dict[Text, List[Features]]]]],
    expected_entity_tags: Optional[List[List[Dict[Text, List[Features]]]]],
) -> bool:
    if (
        actual_features is None
        or actual_labels is None
        or len(actual_features) != len(expected_features)
        or len(actual_labels) != len(expected_labels)
    ):
        return False

    for actual, expected in zip(actual_features, expected_features):
        if not compare_featurized_states(actual, expected):
            return False

    for actual, expected in zip(actual_labels, expected_labels):
        if not np.all(actual == expected):
            return False

    # moodbot doesn't contain e2e entities  # ToDo: Check
    if any([any(turn_tags) for turn_tags in actual_entity_tags]):
        return False
    return True


class TestTrackerFeaturizer:

    TRACKER_FEATURIZER_CLASS = TrackerFeaturizer
    SINGLE_STATE_FEATURIZER_CLASS = SingleStateFeaturizer

    @pytest.fixture
    def moodbot_features(
        self, moodbot_domain: Domain
    ) -> Dict[Text, Dict[Text, Features]]:
        """Creates intent and action features for the moodbot domain.

        Args:
            moodbot_domain: The domain fixture of Moodbot

        Returns:
            Mappings for action and intent names to features.
        """
        origin = self.SINGLE_STATE_FEATURIZER_CLASS.__name__
        action_shape = (1, len(moodbot_domain.action_names_or_texts))
        actions = {}
        for index, action in enumerate(moodbot_domain.action_names_or_texts):
            actions[action] = Features(
                sparse.coo_matrix(([1.0], [[0], [index]]), shape=action_shape),
                FEATURE_TYPE_SENTENCE,
                ACTION_NAME,
                origin,
            )
        intent_shape = (1, len(moodbot_domain.intents))
        intents = {}
        for index, intent in enumerate(moodbot_domain.intents):
            intents[intent] = Features(
                sparse.coo_matrix(([1.0], [[0], [index]]), shape=intent_shape),
                FEATURE_TYPE_SENTENCE,
                INTENT,
                origin,
            )
        return {"intents": intents, "actions": actions}

    @pytest.fixture
    def moodbot_events_with_3_action_unlikely_intent(self) -> List[Event]:
        return [
            ActionExecuted(ACTION_LISTEN_NAME),
            tests.core.utilities.user_uttered("greet"),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ActionExecuted("utter_greet"),
            ActionExecuted(ACTION_LISTEN_NAME),
            tests.core.utilities.user_uttered("mood_unhappy"),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ActionExecuted("utter_cheer_up"),
            ActionExecuted("utter_did_that_help"),
            ActionExecuted(ACTION_LISTEN_NAME),
            tests.core.utilities.user_uttered("deny"),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ActionExecuted("utter_goodbye"),
        ]

    @pytest.fixture
    def moodbot_tracker_with_3_action_unlikely_intent(
        self,
        moodbot_domain: Domain,
        moodbot_events_with_3_action_unlikely_intent: List[Event],
    ) -> DialogueStateTracker:
        return DialogueStateTracker.from_events(
            sender_id="default",
            evts=moodbot_events_with_3_action_unlikely_intent,
            domain=moodbot_domain,
        )

    def test_fail_to_load_non_existent_featurizer(self):
        assert self.TRACKER_FEATURIZER_CLASS.load("non_existent_class") is None

    def test_featurize_trackers_raises_on_missing_state_featurizer(
        self, domain: Domain
    ):
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS()

        with pytest.raises(ValueError):
            tracker_featurizer.featurize_trackers([], domain, RegexInterpreter())

    def test_persist_and_load_tracker_featurizer(
        self, tmp_path: Text, moodbot_domain: Domain
    ):
        state_featurizer = self.SINGLE_STATE_FEATURIZER_CLASS()
        state_featurizer.prepare_for_training(moodbot_domain, RegexInterpreter())
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS(state_featurizer)

        tracker_featurizer.persist(tmp_path)

        loaded_tracker_featurizer = TrackerFeaturizer.load(tmp_path)

        assert loaded_tracker_featurizer is not None
        assert loaded_tracker_featurizer.state_featurizer is not None

    def test__convert_labels_to_ids(self, domain: Domain):
        trackers_as_actions = [
            ["utter_greet", "utter_channel"],
            ["utter_greet", "utter_default", "utter_goodbye"],
        ]

        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS()

        actual_output = tracker_featurizer._convert_labels_to_ids(
            trackers_as_actions, domain
        )
        expected_output = np.array(
            [
                np.array(
                    [
                        domain.action_names_or_texts.index("utter_greet"),
                        domain.action_names_or_texts.index("utter_channel"),
                    ],
                ),
                np.array(
                    [
                        domain.action_names_or_texts.index("utter_greet"),
                        domain.action_names_or_texts.index("utter_default"),
                        domain.action_names_or_texts.index("utter_goodbye"),
                    ],
                ),
            ],
        )

        assert expected_output.size == actual_output.size
        for expected_array, actual_array in zip(expected_output, actual_output):
            assert np.all(expected_array == actual_array)

    @pytest.mark.parametrize(
        "ignore_action_unlikely_intent, max_history, expected_num_action_unlikely_intent",
        [(True, None, 0), (True, 2, 0), (False, None, 3), (False, 2, 3)],
    )
    def test_create_state_features_ignore_action_unlikely_intent(
        self,
        moodbot_domain: Domain,
        moodbot_features: Dict[Text, Dict[Text, Features]],
        moodbot_tracker_with_3_action_unlikely_intent: DialogueStateTracker,
        moodbot_action_unlikely_intent_state_features: Dict[Text, Dict[Text, Features]],
        ignore_action_unlikely_intent: bool,
        max_history: Optional[int],
        expected_num_action_unlikely_intent: int
    ):
        # The base class doesn't implement `create_state_features`
        if self.TRACKER_FEATURIZER_CLASS == TrackerFeaturizer:
            return

        state_featurizer = self.SINGLE_STATE_FEATURIZER_CLASS()
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS(
            state_featurizer, max_history=max_history
        )

        interpreter = RegexInterpreter()
        state_featurizer.prepare_for_training(moodbot_domain, interpreter)
        actual_features = tracker_featurizer.create_state_features(
            [moodbot_tracker_with_3_action_unlikely_intent],
            moodbot_domain,
            interpreter,
            ignore_action_unlikely_intent=ignore_action_unlikely_intent,
        )

        num_action_unlikely_intent_features = count_features(
            actual_features, moodbot_action_unlikely_intent_state_features,
        )

        assert num_action_unlikely_intent_features == expected_num_action_unlikely_intent

    @pytest.mark.parametrize("ignore_action_unlikely_intent", [True, False])
    def test_prediction_states_ignore_action_unlikely_intent(
        self, moodbot_domain: Domain, ignore_action_unlikely_intent: bool
    ):
        # The base class doesn't implement `prediction_states`
        if self.TRACKER_FEATURIZER_CLASS == TrackerFeaturizer:
            return

        state_featurizer = self.SINGLE_STATE_FEATURIZER_CLASS()
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS(state_featurizer)

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

        actual_states = tracker_featurizer.prediction_states(
            [tracker],
            moodbot_domain,
            ignore_action_unlikely_intent=ignore_action_unlikely_intent,
        )

        expected_states = [
            [
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
        ]

        if ignore_action_unlikely_intent:
            # Drop all action_unlikely_intent states
            expected_states = [[expected_states[0][i] for i in {0, 1, 3, 4, 6, 7}]]

        assert actual_states is not None
        assert len(actual_states) == len(expected_states)

        for actual, expected in zip(actual_states, expected_states):
            assert actual == expected


class TestFullDialogueTrackerFeaturizer(TestTrackerFeaturizer):

    TRACKER_FEATURIZER_CLASS = FullDialogueTrackerFeaturizer

    @pytest.mark.parametrize(
        "ignore_action_unlikely_intent", [True, False],
    )
    def test_trackers_ignore_action_unlikely_intent(
        self,
        moodbot_domain: Domain,
        moodbot_features: Dict[Text, Dict[Text, Features]],
        moodbot_events_with_3_action_unlikely_intent: List[Event],
        moodbot_tracker_features_with_3_action_unlikely_intent: List[
            List[Dict[Text, List[Features]]]
        ],
        moodbot_tracker_features_without_action_unlikely_intent: List[
            List[Dict[Text, List[Features]]]
        ],
        moodbot_tracker_with_3_action_unlikely_intent: DialogueStateTracker,
        ignore_action_unlikely_intent: bool,
    ):
        tracker = moodbot_tracker_with_3_action_unlikely_intent
        state_featurizer = self.SINGLE_STATE_FEATURIZER_CLASS()
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS(state_featurizer)

        ignore_action_unlikely_intent = True

        (
            actual_features,
            actual_labels,
            entity_tags,
        ) = tracker_featurizer.featurize_trackers(
            [tracker],
            moodbot_domain,
            RegexInterpreter(),
            ignore_action_unlikely_intent=ignore_action_unlikely_intent,
        )

        expected_features = (
            moodbot_tracker_features_without_action_unlikely_intent
            if ignore_action_unlikely_intent
            else moodbot_tracker_features_with_3_action_unlikely_intent
        )
        expected_labels = [
            moodbot_domain.index_for_action(ACTION_LISTEN_NAME),
            moodbot_domain.index_for_action(ACTION_UNLIKELY_INTENT_NAME),
            moodbot_domain.index_for_action("utter_greet"),
            moodbot_domain.index_for_action(ACTION_LISTEN_NAME),
            moodbot_domain.index_for_action(ACTION_UNLIKELY_INTENT_NAME),
            moodbot_domain.index_for_action("utter_cheer_up"),
            moodbot_domain.index_for_action("utter_did_that_help"),
            moodbot_domain.index_for_action(ACTION_LISTEN_NAME),
            moodbot_domain.index_for_action(ACTION_UNLIKELY_INTENT_NAME),
            moodbot_domain.index_for_action("utter_goodbye"),
        ]

        if ignore_action_unlikely_intent:
            expected_labels = [
                label
                for label in expected_labels
                if label != moodbot_domain.index_for_action(ACTION_UNLIKELY_INTENT_NAME)
            ]

        expected_labels = np.array([expected_labels])

        assert compare_features_and_labels(
            actual_features,
            expected_features,
            actual_labels,
            expected_labels,
            entity_tags,
            None,
        )

    def test_create_state_features(
        self,
        moodbot_tracker: DialogueStateTracker,
        moodbot_domain: Domain,
        moodbot_features: Dict[Text, Dict[Text, Features]],
    ):
        state_featurizer = self.SINGLE_STATE_FEATURIZER_CLASS()
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS(state_featurizer)
        interpreter = RegexInterpreter()
        state_featurizer.prepare_for_training(moodbot_domain, interpreter)
        actual_features = tracker_featurizer.create_state_features(
            [moodbot_tracker], moodbot_domain, interpreter
        )

        expected_features = [
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
                {ACTION_NAME: [moodbot_features["actions"]["utter_goodbye"]]},
            ]
        ]

        assert actual_features is not None
        assert len(actual_features) == len(expected_features)

        for actual, expected in zip(actual_features, expected_features):
            assert compare_featurized_states(actual, expected)

    def test_featurize_trackers(
        self,
        moodbot_tracker: DialogueStateTracker,
        moodbot_domain: Domain,
        moodbot_features: Dict[Text, Dict[Text, Features]],
        moodbot_tracker_features_without_action_unlikely_intent: List[
            List[Dict[Text, List[Features]]]
        ],
    ):
        state_featurizer = self.SINGLE_STATE_FEATURIZER_CLASS()
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS(state_featurizer)

        (
            actual_features,
            actual_labels,
            actual_entity_tags,
        ) = tracker_featurizer.featurize_trackers(
            [moodbot_tracker], moodbot_domain, RegexInterpreter()
        )

        expected_features = moodbot_tracker_features_without_action_unlikely_intent
        expected_labels = np.array(
            [
                [
                    moodbot_domain.index_for_action(ACTION_LISTEN_NAME),
                    moodbot_domain.index_for_action("utter_greet"),
                    moodbot_domain.index_for_action(ACTION_LISTEN_NAME),
                    moodbot_domain.index_for_action("utter_cheer_up"),
                    moodbot_domain.index_for_action("utter_did_that_help"),
                    moodbot_domain.index_for_action(ACTION_LISTEN_NAME),
                    moodbot_domain.index_for_action("utter_goodbye"),
                ]
            ]
        )
        # expected_labels = np.array([
        #     list(map(moodbot_domain.index_for_action, [
        #         ACTION_LISTEN_NAME,
        #         "utter_greet",
        #         ACTION_LISTEN_NAME,
        #         "utter_cheer_up",
        #         "utter_did_that_help",
        #         ACTION_LISTEN_NAME,
        #         "utter_goodbye",
        #     ]))
        # ])

        assert compare_features_and_labels(
            actual_features,
            expected_features,
            actual_labels,
            expected_labels,
            actual_entity_tags,
            expected_entity_tags=None,
        )

    def test_prediction_states(
        self, moodbot_tracker: DialogueStateTracker, moodbot_domain: Domain
    ):
        state_featurizer = self.SINGLE_STATE_FEATURIZER_CLASS()
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS(state_featurizer)
        actual_states = tracker_featurizer.prediction_states(
            [moodbot_tracker], moodbot_domain,
        )

        expected_states = [
            [
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
        ]

        assert actual_states is not None
        assert len(actual_states) == len(expected_states)

        for actual, expected in zip(actual_states, expected_states):
            assert actual == expected

    def test_prediction_states_hide_rule_states_with_full_dialogue_tracker_featurizer(
        self, moodbot_domain: Domain,
    ):
        state_featurizer = self.SINGLE_STATE_FEATURIZER_CLASS()
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS(state_featurizer)

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

        actual_states = tracker_featurizer.prediction_states(
            [rule_tracker], moodbot_domain, ignore_rule_only_turns=True,
        )

        expected_states = [
            [
                {},
                {
                    PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
                    USER: {INTENT: "greet"},
                },
            ],
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

        actual_states = tracker_featurizer.prediction_states(
            [embedded_rule_tracker], moodbot_domain, ignore_rule_only_turns=True,
        )

        expected_states = [
            [
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
                    USER: {INTENT: "mood_great"},
                    PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
                },
            ]
        ]

        assert actual_states is not None
        assert len(actual_states) == len(expected_states)

        for actual, expected in zip(actual_states, expected_states):
            assert actual == expected


class TestMaxHistoryTrackerFeaturizer(TestTrackerFeaturizer):

    TRACKER_FEATURIZER_CLASS = MaxHistoryTrackerFeaturizer

    @pytest.mark.parametrize(
        "ignore_action_unlikely_intent, max_history, expected_num_action_unlikely_intent",
        [
            # Same as in base class
            (True, None, 0), (True, 2, 0), (False, None, 3),
            # In this case, `expected_num_action_unlikely_intent` is 1 instead of 3,
            # because for max-history featurizers only one `action_unlikely_intent`
            # is left after slicing.
            (False, 2, 1)
        ],
    )
    def test_create_state_features_ignore_action_unlikely_intent(
        self,
        moodbot_domain: Domain,
        moodbot_features: Dict[Text, Dict[Text, Features]],
        moodbot_tracker_with_3_action_unlikely_intent: DialogueStateTracker,
        moodbot_action_unlikely_intent_state_features: Dict[Text, Dict[Text, Features]],
        ignore_action_unlikely_intent: bool,
        max_history: Optional[int],
        expected_num_action_unlikely_intent: int
    ):
        super().test_create_state_features_ignore_action_unlikely_intent(
            moodbot_domain,
            moodbot_features,
            moodbot_tracker_with_3_action_unlikely_intent,
            moodbot_action_unlikely_intent_state_features,
            ignore_action_unlikely_intent,
            max_history,
            expected_num_action_unlikely_intent,
        )


class TestIntentMaxHistoryTrackerFeaturizer(TestMaxHistoryTrackerFeaturizer):

    TRACKER_FEATURIZER_CLASS = IntentMaxHistoryTrackerFeaturizer
    SINGLE_STATE_FEATURIZER_CLASS = IntentTokenizerSingleStateFeaturizer

    def test__convert_labels_to_ids(self, domain: Domain):
        """.  # ToDo

        Overwritten, because the labels are intents now.
        """
        trackers_as_intents = [
            ["next_intent", "nlu_fallback", "out_of_scope", "restart"],
            ["greet", "hello", "affirm"],
        ]

        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS()

        actual_labels = tracker_featurizer._convert_labels_to_ids(
            trackers_as_intents, domain
        )

        expected_labels = np.array(
            [
                [
                    domain.intents.index("next_intent"),
                    domain.intents.index("nlu_fallback"),
                    domain.intents.index("out_of_scope"),
                    domain.intents.index("restart"),
                ],
                [
                    domain.intents.index("greet"),
                    domain.intents.index("hello"),
                    domain.intents.index("affirm"),
                    LABEL_PAD_ID,
                ],
            ],
        )
        assert expected_labels.size == actual_labels.size
        assert expected_labels.shape == actual_labels.shape
        assert np.all(expected_labels == actual_labels)

    @pytest.mark.parametrize("ignore_action_unlikely_intent", [True, False])
    def test_prediction_states_ignore_action_unlikely_intent(
        self, moodbot_domain: Domain, ignore_action_unlikely_intent: bool
    ):
        """.

        Overwritten because last user utterance should be removed.
        """

        state_featurizer = self.SINGLE_STATE_FEATURIZER_CLASS()
        tracker_featurizer = self.TRACKER_FEATURIZER_CLASS(state_featurizer)

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

        actual_states = tracker_featurizer.prediction_states(
            [tracker],
            moodbot_domain,
            ignore_action_unlikely_intent=ignore_action_unlikely_intent,
        )

        expected_states = [
            [
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
                # {
                #     PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
                #     USER: {INTENT: "goodbye"},
                # },
            ]
        ]

        if ignore_action_unlikely_intent:
            # Drop all action_unlikely_intent states
            expected_states = [[expected_states[0][i] for i in {0, 1, 3, 4, 6}]]

        assert actual_states is not None
        assert len(actual_states) == len(expected_states)

        for actual, expected in zip(actual_states, expected_states):
            assert actual == expected


# def test_state_features_keep_action_unlikely_intent_full_dialogue_tracker_featurizer(
#     moodbot_domain: Domain, moodbot_features: Dict[Text, Dict[Text, Features]],
# ):
#     tracker = DialogueStateTracker.from_events(
#         "default",
#         [
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("greet"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_greet"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("mood_great"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_happy"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("goodbye"),
#         ],
#         domain=moodbot_domain,
#     )

#     state_featurizer = SingleStateFeaturizer()
#     tracker_featurizer = FullDialogueTrackerFeaturizer(state_featurizer)
#     interpreter = RegexInterpreter()
#     state_featurizer.prepare_for_training(moodbot_domain, interpreter)
#     actual_features = tracker_featurizer.create_state_features(
#         [tracker], moodbot_domain, interpreter,
#     )

#     expected_features = [
#         [
#             {},
#             {
#                 ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
#                 INTENT: [moodbot_features["intents"]["greet"]],
#             },
#             {ACTION_NAME: [moodbot_features["actions"][ACTION_UNLIKELY_INTENT_NAME]]},
#             {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
#             {
#                 ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
#                 INTENT: [moodbot_features["intents"]["mood_great"]],
#             },
#             {ACTION_NAME: [moodbot_features["actions"][ACTION_UNLIKELY_INTENT_NAME]]},
#             {ACTION_NAME: [moodbot_features["actions"]["utter_happy"]]},
#             {
#                 ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
#                 INTENT: [moodbot_features["intents"]["goodbye"]],
#             },
#         ]
#     ]

#     assert actual_features is not None
#     assert len(actual_features) == len(expected_features)

#     for actual, expected in zip(actual_features, expected_features):
#         assert compare_featurized_states(actual, expected)


# def test_prediction_states_with_full_dialogue_tracker_featurizer(
#     moodbot_tracker: DialogueStateTracker, moodbot_domain: Domain
# ):
#     state_featurizer = SingleStateFeaturizer()
#     tracker_featurizer = FullDialogueTrackerFeaturizer(state_featurizer)
#     actual_states = tracker_featurizer.prediction_states(
#         [moodbot_tracker], moodbot_domain,
#     )

#     expected_states = [
#         [
#             {},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "greet"},
#             },
#             {USER: {INTENT: "greet"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "mood_unhappy"},
#             },
#             {
#                 USER: {INTENT: "mood_unhappy"},
#                 PREVIOUS_ACTION: {ACTION_NAME: "utter_cheer_up"},
#             },
#             {
#                 USER: {INTENT: "mood_unhappy"},
#                 PREVIOUS_ACTION: {ACTION_NAME: "utter_did_that_help"},
#             },
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "deny"},
#             },
#             {USER: {INTENT: "deny"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_goodbye"},},
#         ]
#     ]

#     assert actual_states is not None
#     assert len(actual_states) == len(expected_states)

#     for actual, expected in zip(actual_states, expected_states):
#         assert actual == expected


# def test_prediction_states_hide_rule_states_with_full_dialogue_tracker_featurizer(
#     moodbot_domain: Domain,
# ):

#     state_featurizer = SingleStateFeaturizer()
#     tracker_featurizer = FullDialogueTrackerFeaturizer(state_featurizer)

#     rule_tracker = DialogueStateTracker.from_events(
#         "default",
#         [
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("greet"),
#             ActionExecuted("utter_greet", hide_rule_turn=True),
#             ActionExecuted(ACTION_LISTEN_NAME, hide_rule_turn=True),
#         ],
#         domain=moodbot_domain,
#     )

#     actual_states = tracker_featurizer.prediction_states(
#         [rule_tracker], moodbot_domain, ignore_rule_only_turns=True,
#     )

#     expected_states = [
#         [
#             {},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "greet"},
#             },
#         ],
#     ]

#     assert actual_states is not None
#     assert len(actual_states) == len(expected_states)

#     for actual, expected in zip(actual_states, expected_states):
#         assert actual == expected

#     embedded_rule_tracker = DialogueStateTracker.from_events(
#         "default",
#         [
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("greet"),
#             ActionExecuted("utter_greet", hide_rule_turn=True),
#             ActionExecuted(ACTION_LISTEN_NAME, hide_rule_turn=True),
#             user_uttered("mood_great"),
#             ActionExecuted("utter_happy"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#         ],
#         domain=moodbot_domain,
#     )

#     actual_states = tracker_featurizer.prediction_states(
#         [embedded_rule_tracker], moodbot_domain, ignore_rule_only_turns=True,
#     )

#     expected_states = [
#         [
#             {},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "mood_great"},
#             },
#             {
#                 USER: {INTENT: "mood_great"},
#                 PREVIOUS_ACTION: {ACTION_NAME: "utter_happy"},
#             },
#             {
#                 USER: {INTENT: "mood_great"},
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#             },
#         ]
#     ]

#     assert actual_states is not None
#     assert len(actual_states) == len(expected_states)

#     for actual, expected in zip(actual_states, expected_states):
#         assert actual == expected


# def test_prediction_states_ignore_action_intent_unlikely_full_dialogue_featurizer(
#     moodbot_domain: Domain,
# ):
#     state_featurizer = SingleStateFeaturizer()
#     tracker_featurizer = FullDialogueTrackerFeaturizer(state_featurizer)

#     tracker = DialogueStateTracker.from_events(
#         "default",
#         [
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("greet"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_greet"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("mood_great"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_happy"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("goodbye"),
#         ],
#         domain=moodbot_domain,
#     )

#     actual_states = tracker_featurizer.prediction_states(
#         [tracker], moodbot_domain, ignore_action_unlikely_intent=True
#     )

#     expected_states = [
#         [
#             {},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "greet"},
#             },
#             {USER: {INTENT: "greet"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "mood_great"},
#             },
#             {
#                 USER: {INTENT: "mood_great"},
#                 PREVIOUS_ACTION: {ACTION_NAME: "utter_happy"},
#             },
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "goodbye"},
#             },
#         ]
#     ]

#     assert actual_states is not None
#     assert len(actual_states) == len(expected_states)

#     for actual, expected in zip(actual_states, expected_states):
#         assert actual == expected


# def test_prediction_states_keeps_action_intent_unlikely_full_dialogue_featurizer(
#     moodbot_domain: Domain,
# ):

#     state_featurizer = SingleStateFeaturizer()
#     tracker_featurizer = FullDialogueTrackerFeaturizer(state_featurizer)

#     tracker = DialogueStateTracker.from_events(
#         "default",
#         [
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("greet"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_greet"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("mood_great"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_happy"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("goodbye"),
#         ],
#         domain=moodbot_domain,
#     )

#     actual_states = tracker_featurizer.prediction_states([tracker], moodbot_domain,)

#     expected_states = [
#         [
#             {},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "greet"},
#             },
#             {
#                 USER: {INTENT: "greet"},
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_UNLIKELY_INTENT_NAME},
#             },
#             {USER: {INTENT: "greet"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "mood_great"},
#             },
#             {
#                 USER: {INTENT: "mood_great"},
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_UNLIKELY_INTENT_NAME},
#             },
#             {
#                 USER: {INTENT: "mood_great"},
#                 PREVIOUS_ACTION: {ACTION_NAME: "utter_happy"},
#             },
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "goodbye"},
#             },
#         ]
#     ]

#     assert actual_states is not None
#     assert len(actual_states) == len(expected_states)

#     for actual, expected in zip(actual_states, expected_states):
#         assert actual == expected


@pytest.mark.parametrize("max_history", [None, 2])
def test_featurize_trackers_with_max_history_tracker_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
    )

    actual_features, actual_labels, entity_tags = tracker_featurizer.featurize_trackers(
        [moodbot_tracker], moodbot_domain, RegexInterpreter()
    )

    expected_features = [
        [{},],
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

    expected_labels = np.array([[0, 15, 0, 12, 13, 0, 14]]).T

    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape
    assert np.all(actual_labels == expected_labels)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_featurize_trackers_ignore_action_unlikely_intent_max_history_featurizer(
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
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history,
    )

    actual_features, actual_labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker],
        moodbot_domain,
        RegexInterpreter(),
        ignore_action_unlikely_intent=True,
    )

    expected_features = [
        [{},],
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

    expected_labels = np.array([[0, 15, 0]]).T
    assert actual_labels.shape == expected_labels.shape
    for actual, expected in zip(actual_labels, expected_labels):
        assert np.all(actual == expected)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_featurize_trackers_keep_action_unlikely_intent_max_history_featurizer(
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
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history,
    )

    actual_features, actual_labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker], moodbot_domain, RegexInterpreter(),
    )

    expected_features = [
        [{},],
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

    expected_labels = np.array([[0, 9, 15, 0]]).T
    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape
    for actual, expected in zip(actual_labels, expected_labels):
        assert np.all(actual == expected)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize(
    "remove_duplicates,max_history",
    [[True, None], [True, 2], [False, None], [False, 2],],
)
def test_deduplicate_featurize_trackers_with_max_history_tracker_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    remove_duplicates: bool,
    max_history: Optional[int],
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history, remove_duplicates=remove_duplicates
    )

    # Add Duplicate moodbot_tracker states should get removed.
    actual_features, actual_labels, entity_tags = tracker_featurizer.featurize_trackers(
        [moodbot_tracker, moodbot_tracker], moodbot_domain, RegexInterpreter()
    )

    expected_features = [
        [{},],
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

    expected_labels = np.array([[0, 15, 0, 12, 13, 0, 14]]).T
    if not remove_duplicates:
        expected_labels = np.vstack([expected_labels] * 2)

    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape
    assert np.all(actual_labels == expected_labels)

    # moodbot doesn't contain e2e entities
    assert not any([any(turn_tags) for turn_tags in entity_tags])


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_state_features_with_max_history_tracker_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):
    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
    )
    interpreter = RegexInterpreter()
    state_featurizer.prepare_for_training(moodbot_domain, interpreter)
    actual_features = tracker_featurizer.create_state_features(
        [moodbot_tracker], moodbot_domain, interpreter
    )

    expected_features = [
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
            {ACTION_NAME: [moodbot_features["actions"]["utter_goodbye"]]},
        ]
    ]
    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]

    assert actual_features is not None
    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_state_features_ignore_action_unlikely_intent_max_history_featurizer(
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

    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
    )
    interpreter = RegexInterpreter()
    state_featurizer.prepare_for_training(moodbot_domain, interpreter)
    actual_features = tracker_featurizer.create_state_features(
        [tracker], moodbot_domain, interpreter, ignore_action_unlikely_intent=True
    )

    expected_features = [
        [
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
    ]
    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]

    assert actual_features is not None
    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)


@pytest.mark.parametrize("max_history", [None, 2])
def test_create_state_features_keep_action_unlikely_intent_max_history_featurizer(
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

    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
    )
    interpreter = RegexInterpreter()
    state_featurizer.prepare_for_training(moodbot_domain, interpreter)
    actual_features = tracker_featurizer.create_state_features(
        [tracker], moodbot_domain, interpreter,
    )

    expected_features = [
        [
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
    ]
    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]

    assert actual_features is not None
    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)


# @pytest.mark.parametrize("max_history", [None, 2])
# def test_prediction_states_with_max_history_tracker_featurizer(
#     moodbot_tracker: DialogueStateTracker,
#     moodbot_domain: Domain,
#     max_history: Optional[int],
# ):

#     state_featurizer = SingleStateFeaturizer()
#     tracker_featurizer = MaxHistoryTrackerFeaturizer(
#         state_featurizer, max_history=max_history
#     )
#     actual_states = tracker_featurizer.prediction_states(
#         [moodbot_tracker], moodbot_domain,
#     )

#     expected_states = [
#         [
#             {},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "greet"},
#             },
#             {USER: {INTENT: "greet"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "mood_unhappy"},
#             },
#             {
#                 USER: {INTENT: "mood_unhappy"},
#                 PREVIOUS_ACTION: {ACTION_NAME: "utter_cheer_up"},
#             },
#             {
#                 USER: {INTENT: "mood_unhappy"},
#                 PREVIOUS_ACTION: {ACTION_NAME: "utter_did_that_help"},
#             },
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "deny"},
#             },
#             {USER: {INTENT: "deny"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_goodbye"},},
#         ]
#     ]
#     if max_history is not None:
#         expected_states = [x[-max_history:] for x in expected_states]

#     assert actual_states is not None
#     assert len(actual_states) == len(expected_states)

#     for actual, expected in zip(actual_states, expected_states):
#         assert actual == expected


@pytest.mark.parametrize("max_history", [None, 2])
def test_prediction_states_hide_rule_states_with_max_history_tracker_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):

    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
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

    actual_states = tracker_featurizer.prediction_states(
        [rule_tracker], moodbot_domain, ignore_rule_only_turns=True,
    )

    expected_states = [
        [
            {},
            {
                PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
                USER: {INTENT: "greet"},
            },
        ],
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

    actual_states = tracker_featurizer.prediction_states(
        [embedded_rule_tracker], moodbot_domain, ignore_rule_only_turns=True,
    )

    expected_states = [
        [
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
    ]

    if max_history is not None:
        expected_states = [x[-max_history:] for x in expected_states]

    assert actual_states is not None
    assert len(actual_states) == len(expected_states)

    for actual, expected in zip(actual_states, expected_states):
        assert actual == expected


@pytest.mark.parametrize("max_history", [None, 3])
def test_prediction_states_ignores_action_intent_unlikely_max_history_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):

    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
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

    actual_states = tracker_featurizer.prediction_states(
        [tracker], moodbot_domain, ignore_action_unlikely_intent=True
    )

    expected_states = [
        [
            {},
            {
                PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
                USER: {INTENT: "greet"},
            },
            {USER: {INTENT: "greet"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},},
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
    ]

    if max_history is not None:
        expected_states = [x[-max_history:] for x in expected_states]

    assert actual_states is not None
    assert len(actual_states) == len(expected_states)

    for actual, expected in zip(actual_states, expected_states):
        assert actual == expected


@pytest.mark.parametrize("max_history", [None, 3])
def test_prediction_states_keeps_action_intent_unlikely_max_history_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):

    state_featurizer = SingleStateFeaturizer()
    tracker_featurizer = MaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
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

    actual_states = tracker_featurizer.prediction_states([tracker], moodbot_domain,)

    expected_states = [
        [
            {},
            {
                PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
                USER: {INTENT: "greet"},
            },
            {
                USER: {INTENT: "greet"},
                PREVIOUS_ACTION: {ACTION_NAME: ACTION_UNLIKELY_INTENT_NAME},
            },
            {USER: {INTENT: "greet"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},},
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
    ]

    if max_history is not None:
        expected_states = [x[-max_history:] for x in expected_states]

    assert actual_states is not None
    assert len(actual_states) == len(expected_states)

    for actual, expected in zip(actual_states, expected_states):
        assert actual == expected


@pytest.mark.parametrize(
    "max_history,moodbot_features",
    [
        [None, "IntentTokenizerSingleStateFeaturizer"],
        [2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
def test_featurize_trackers_with_intent_max_history_tracker_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    max_history: Optional[int],
):
    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
    )

    actual_features, actual_labels, entity_tags = tracker_featurizer.featurize_trackers(
        [moodbot_tracker], moodbot_domain, RegexInterpreter()
    )

    expected_features = [
        [{},],
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
def test_trackers_ignore_action_unlikely_intent_intent_max_history_featurizer(
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
    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history,
    )

    actual_features, actual_labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker],
        moodbot_domain,
        RegexInterpreter(),
        ignore_action_unlikely_intent=True,
    )

    expected_features = [
        [{},],
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
def test_trackers_keep_action_unlikely_intent_intent_max_history_featurizer(
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
    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history,
    )

    actual_features, actual_labels, entity_tags = tracker_featurizer.featurize_trackers(
        [tracker], moodbot_domain, RegexInterpreter(),
    )

    expected_features = [
        [{},],
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
def test_deduplicate_featurize_trackers_with_intent_max_history_tracker_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    moodbot_features: Dict[Text, Dict[Text, Features]],
    remove_duplicates: bool,
    max_history: Optional[int],
):
    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history, remove_duplicates=remove_duplicates
    )

    # Add Duplicate moodbot_tracker states should get removed.
    actual_features, actual_labels, entity_tags = tracker_featurizer.featurize_trackers(
        [moodbot_tracker, moodbot_tracker], moodbot_domain, RegexInterpreter()
    )

    expected_features = [
        [{},],
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
    "max_history,moodbot_features",
    [
        [None, "IntentTokenizerSingleStateFeaturizer"],
        [2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
def test_create_state_features_with_intent_max_history_tracker_featurizer(
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

    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
    )
    interpreter = RegexInterpreter()
    state_featurizer.prepare_for_training(moodbot_domain, interpreter)
    actual_features = tracker_featurizer.create_state_features(
        [moodbot_tracker], moodbot_domain, interpreter
    )

    expected_features = [
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


# @pytest.mark.parametrize(
#     "max_history,moodbot_features",
#     [
#         [None, "IntentTokenizerSingleStateFeaturizer"],
#         [2, "IntentTokenizerSingleStateFeaturizer"],
#     ],
#     indirect=["moodbot_features"],
# )
# def test_state_features_ignore_action_unlikely_intent_intent_max_history_featurizer(
#     moodbot_domain: Domain,
#     moodbot_features: Dict[Text, Dict[Text, Features]],
#     max_history: Optional[int],
# ):

#     tracker = DialogueStateTracker.from_events(
#         "default",
#         [
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("greet"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_greet"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("mood_great"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_happy"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("goodbye"),
#         ],
#         domain=moodbot_domain,
#     )
#     state_featurizer = IntentTokenizerSingleStateFeaturizer()
#     tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
#         state_featurizer, max_history=max_history
#     )
#     interpreter = RegexInterpreter()
#     state_featurizer.prepare_for_training(moodbot_domain, interpreter)
#     actual_features = tracker_featurizer.create_state_features(
#         [tracker], moodbot_domain, interpreter, ignore_action_unlikely_intent=True,
#     )

#     expected_features = [
#         [
#             {},
#             {
#                 ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
#                 INTENT: [moodbot_features["intents"]["greet"]],
#             },
#             {ACTION_NAME: [moodbot_features["actions"]["utter_greet"]]},
#             {
#                 ACTION_NAME: [moodbot_features["actions"][ACTION_LISTEN_NAME]],
#                 INTENT: [moodbot_features["intents"]["mood_great"]],
#             },
#             {ACTION_NAME: [moodbot_features["actions"]["utter_happy"]]},
#         ],
#     ]
#     if max_history is not None:
#         expected_features = [x[-max_history:] for x in expected_features]

#     assert actual_features is not None
#     assert len(actual_features) == len(expected_features)

#     for actual, expected in zip(actual_features, expected_features):
#         assert compare_featurized_states(actual, expected)


@pytest.mark.parametrize(
    "max_history,moodbot_features",
    [
        [None, "IntentTokenizerSingleStateFeaturizer"],
        [2, "IntentTokenizerSingleStateFeaturizer"],
    ],
    indirect=["moodbot_features"],
)
def test_state_features_keep_action_unlikely_intent_intent_max_history_featurizer(
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
    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
    )
    interpreter = RegexInterpreter()
    state_featurizer.prepare_for_training(moodbot_domain, interpreter)
    actual_features = tracker_featurizer.create_state_features(
        [tracker], moodbot_domain, interpreter,
    )

    expected_features = [
        [
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
        ],
    ]

    if max_history is not None:
        expected_features = [x[-max_history:] for x in expected_features]

    assert actual_features is not None
    assert len(actual_features) == len(expected_features)

    for actual, expected in zip(actual_features, expected_features):
        assert compare_featurized_states(actual, expected)


@pytest.mark.parametrize("max_history", [None, 2])
def test_prediction_states_with_intent_max_history_tracker_featurizer(
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

    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
    )
    actual_states = tracker_featurizer.prediction_states(
        [moodbot_tracker], moodbot_domain,
    )

    expected_states = [
        [
            {},
            {
                PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
                USER: {INTENT: "greet"},
            },
            {USER: {INTENT: "greet"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},},
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
    ]
    if max_history is not None:
        expected_states = [x[-max_history:] for x in expected_states]

    assert actual_states is not None
    assert len(actual_states) == len(expected_states)

    for actual, expected in zip(actual_states, expected_states):
        assert actual == expected


@pytest.mark.parametrize("max_history", [None, 2])
def test_prediction_states_hide_rule_states_intent_max_history_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):

    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
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

    actual_states = tracker_featurizer.prediction_states(
        [rule_tracker], moodbot_domain, ignore_rule_only_turns=True,
    )

    expected_states = [[{}]]

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
        ],
        domain=moodbot_domain,
    )

    actual_states = tracker_featurizer.prediction_states(
        [embedded_rule_tracker], moodbot_domain, ignore_rule_only_turns=True,
    )

    expected_states = [[{},]]

    assert actual_states is not None
    assert len(actual_states) == len(expected_states)

    for actual, expected in zip(actual_states, expected_states):
        assert actual == expected


# @pytest.mark.parametrize("max_history", [None, 3])
# def test_prediction_states_ignores_action_intent_unlikely_intent_max_history_featurizer(
#     moodbot_tracker: DialogueStateTracker,
#     moodbot_domain: Domain,
#     max_history: Optional[int],
# ):

#     state_featurizer = IntentTokenizerSingleStateFeaturizer()
#     tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
#         state_featurizer, max_history=max_history
#     )

#     tracker = DialogueStateTracker.from_events(
#         "default",
#         [
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("greet"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_greet"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("mood_great"),
#             ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
#             ActionExecuted("utter_happy"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             user_uttered("goodbye"),
#         ],
#         domain=moodbot_domain,
#     )

#     actual_states = tracker_featurizer.prediction_states(
#         [tracker], moodbot_domain, ignore_action_unlikely_intent=True
#     )

#     expected_states = [
#         [
#             {},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "greet"},
#             },
#             {USER: {INTENT: "greet"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},},
#             {
#                 PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
#                 USER: {INTENT: "mood_great"},
#             },
#             {
#                 USER: {INTENT: "mood_great"},
#                 PREVIOUS_ACTION: {ACTION_NAME: "utter_happy"},
#             },
#         ]
#     ]

#     if max_history is not None:
#         expected_states = [x[-max_history:] for x in expected_states]

#     assert actual_states is not None
#     assert len(actual_states) == len(expected_states)

#     for actual, expected in zip(actual_states, expected_states):
#         assert actual == expected


@pytest.mark.parametrize("max_history", [None, 3])
def test_prediction_states_keeps_action_intent_unlikely_intent_max_history_featurizer(
    moodbot_tracker: DialogueStateTracker,
    moodbot_domain: Domain,
    max_history: Optional[int],
):

    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history
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

    actual_states = tracker_featurizer.prediction_states([tracker], moodbot_domain)

    expected_states = [
        [
            {},
            {
                PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
                USER: {INTENT: "greet"},
            },
            {
                USER: {INTENT: "greet"},
                PREVIOUS_ACTION: {ACTION_NAME: ACTION_UNLIKELY_INTENT_NAME},
            },
            {USER: {INTENT: "greet"}, PREVIOUS_ACTION: {ACTION_NAME: "utter_greet"},},
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
    ]

    if max_history is not None:
        expected_states = [x[-max_history:] for x in expected_states]

    assert actual_states is not None
    assert len(actual_states) == len(expected_states)

    for actual, expected in zip(actual_states, expected_states):
        assert actual == expected


@pytest.mark.parametrize(
    "remove_duplicates,max_history",
    [[True, None], [True, 2], [False, None], [False, 2],],
)
def test_multilabels_with_intent_max_history_tracker_featurizer(
    moodbot_domain: Domain, max_history: Optional[int], remove_duplicates: bool
):
    state_featurizer = IntentTokenizerSingleStateFeaturizer()
    tracker_featurizer = IntentMaxHistoryTrackerFeaturizer(
        state_featurizer, max_history=max_history, remove_duplicates=remove_duplicates,
    )

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

    _1, actual_labels, _2 = tracker_featurizer.featurize_trackers(
        [tracker1, tracker2], moodbot_domain, RegexInterpreter()
    )

    greet_index = 5
    mood_great_index = 6
    mood_unhappy_index = 7

    if remove_duplicates:
        expected_labels = np.array(
            [
                [greet_index, -1],
                [mood_great_index, mood_unhappy_index],
                [mood_unhappy_index, mood_great_index],
            ]
        )
    else:
        expected_labels = np.array(
            [
                [greet_index, -1],
                [mood_great_index, mood_unhappy_index],
                [greet_index, -1],
                [mood_unhappy_index, mood_great_index],
            ]
        )

    assert actual_labels is not None
    assert actual_labels.shape == expected_labels.shape
    assert np.all(actual_labels == expected_labels)
