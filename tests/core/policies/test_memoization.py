import pytest
from tests.core.test_policies import PolicyTestCollection
from typing import Optional
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    SlotSet,
)
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.nlu.interpreter import RegexInterpreter


class TestMemoizationPolicy(PolicyTestCollection):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> MemoizationPolicy:
        return MemoizationPolicy(featurizer=featurizer, priority=priority)

    @pytest.mark.parametrize("max_history", [1, 2, 3, 4, None])
    def test_prediction(self, max_history):
        policy = self.create_policy(
            featurizer=MaxHistoryTrackerFeaturizer(max_history=max_history), priority=1
        )

        GREET_INTENT_NAME = "greet"
        UTTER_GREET_ACTION = "utter_greet"
        UTTER_BYE_ACTION = "utter_goodbye"
        domain = Domain.from_yaml(
            f"""
            intents:
            - {GREET_INTENT_NAME}
            actions:
            - {UTTER_GREET_ACTION}
            - {UTTER_BYE_ACTION}
            slots:
                slot_1:
                    type: bool
                slot_2:
                    type: bool
                slot_3:
                    type: bool
                slot_4:
                    type: bool
            """
        )
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            SlotSet("slot_1", True),
            ActionExecuted(UTTER_GREET_ACTION),
            SlotSet("slot_2", True),
            SlotSet("slot_3", True),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            SlotSet("slot_4", True),
            ActionExecuted(UTTER_BYE_ACTION),
            ActionExecuted(ACTION_LISTEN_NAME),
        ]
        training_story = TrackerWithCachedStates.from_events(
            "training story", evts=events, domain=domain, slots=domain.slots,
        )
        test_story = TrackerWithCachedStates.from_events(
            "training story", events[:-2], domain=domain, slots=domain.slots,
        )
        interpreter = RegexInterpreter()
        policy.train([training_story], domain, interpreter)
        prediction = policy.predict_action_probabilities(
            test_story, domain, interpreter
        )
        assert (
            domain.action_names_or_texts[
                prediction.probabilities.index(max(prediction.probabilities))
            ]
            == UTTER_BYE_ACTION
        )


class TestAugmentedMemoizationPolicy(TestMemoizationPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> MemoizationPolicy:
        return AugmentedMemoizationPolicy(featurizer=featurizer, priority=priority)

    @pytest.mark.parametrize("max_history", [1, 2, 3, 4, None])
    def test_augmented_prediction(self, max_history):
        policy = self.create_policy(
            featurizer=MaxHistoryTrackerFeaturizer(max_history=max_history), priority=1
        )

        GREET_INTENT_NAME = "greet"
        UTTER_GREET_ACTION = "utter_greet"
        UTTER_BYE_ACTION = "utter_goodbye"
        domain = Domain.from_yaml(
            f"""
            intents:
            - {GREET_INTENT_NAME}
            actions:
            - {UTTER_GREET_ACTION}
            - {UTTER_BYE_ACTION}
            slots:
                slot_1:
                    type: bool
                    initial_value: true
                slot_2:
                    type: bool
                slot_3:
                    type: bool
            """
        )
        training_story = TrackerWithCachedStates.from_events(
            "training story",
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_3", True),
                ActionExecuted(UTTER_BYE_ACTION),
                ActionExecuted(ACTION_LISTEN_NAME),
            ],
            domain=domain,
            slots=domain.slots,
        )
        test_story = TrackerWithCachedStates.from_events(
            "test story",
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_1", False),
                ActionExecuted(UTTER_GREET_ACTION),
                ActionExecuted(UTTER_GREET_ACTION),
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_2", True),
                ActionExecuted(UTTER_GREET_ACTION),
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_3", True),
                # ActionExecuted(UTTER_BYE_ACTION),
            ],
            domain=domain,
            slots=domain.slots,
        )
        interpreter = RegexInterpreter()
        policy.train([training_story], domain, interpreter)
        prediction = policy.predict_action_probabilities(
            test_story, domain, interpreter
        )
        assert (
            domain.action_names_or_texts[
                prediction.probabilities.index(max(prediction.probabilities))
            ]
            == UTTER_BYE_ACTION
        )

    @pytest.mark.parametrize("max_history", [1, 2, 3, 4, None])
    def test_augmented_prediction_across_max_history_actions(self, max_history):
        """Tests that the last user utterance is preserved in action states
        even when the utterance occurs prior to `max_history` actions in the
        past.
        """
        policy = self.create_policy(
            featurizer=MaxHistoryTrackerFeaturizer(max_history=max_history), priority=1
        )

        GREET_INTENT_NAME = "greet"
        UTTER_GREET_ACTION = "utter_greet"
        UTTER_ACTION_1 = "utter_1"
        UTTER_ACTION_2 = "utter_2"
        UTTER_ACTION_3 = "utter_3"
        UTTER_ACTION_4 = "utter_4"
        UTTER_ACTION_5 = "utter_5"
        UTTER_BYE_ACTION = "utter_goodbye"
        domain = Domain.from_yaml(
            f"""
            intents:
            - {GREET_INTENT_NAME}
            actions:
            - {UTTER_GREET_ACTION}
            - {UTTER_ACTION_1}
            - {UTTER_ACTION_2}
            - {UTTER_ACTION_3}
            - {UTTER_ACTION_4}
            - {UTTER_ACTION_5}
            - {UTTER_BYE_ACTION}
            """
        )
        training_story = TrackerWithCachedStates.from_events(
            "training story",
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_ACTION_1),
                ActionExecuted(UTTER_ACTION_2),
                ActionExecuted(UTTER_ACTION_3),
                ActionExecuted(UTTER_ACTION_4),
                ActionExecuted(UTTER_ACTION_5),
                ActionExecuted(UTTER_BYE_ACTION),
                ActionExecuted(ACTION_LISTEN_NAME),
            ],
            domain=domain,
            slots=domain.slots,
        )
        test_story = TrackerWithCachedStates.from_events(
            "test story",
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_ACTION_1),
                ActionExecuted(UTTER_ACTION_2),
                ActionExecuted(UTTER_ACTION_3),
                ActionExecuted(UTTER_ACTION_4),
                ActionExecuted(UTTER_ACTION_5),
                # ActionExecuted(UTTER_BYE_ACTION),
            ],
            domain=domain,
            slots=domain.slots,
        )
        interpreter = RegexInterpreter()
        policy.train([training_story], domain, interpreter)
        prediction = policy.predict_action_probabilities(
            test_story, domain, interpreter
        )
        assert (
            domain.action_names_or_texts[
                prediction.probabilities.index(max(prediction.probabilities))
            ]
            == UTTER_BYE_ACTION
        )

    @pytest.mark.parametrize("max_history", [1, 2, 3, 4, None])
    def test_aug_pred_sensitive_to_intent_across_max_history_actions(self, max_history):
        """Tests that only the most recent user utterance propagates to state
        creation of following actions.
        """
        policy = self.create_policy(
            featurizer=MaxHistoryTrackerFeaturizer(max_history=max_history), priority=1
        )

        GREET_INTENT_NAME = "greet"
        GOODBYE_INTENT_NAME = "goodbye"
        UTTER_GREET_ACTION = "utter_greet"
        UTTER_ACTION_1 = "utter_1"
        UTTER_ACTION_2 = "utter_2"
        UTTER_ACTION_3 = "utter_3"
        UTTER_ACTION_4 = "utter_4"
        UTTER_ACTION_5 = "utter_5"
        UTTER_BYE_ACTION = "utter_goodbye"
        domain = Domain.from_yaml(
            f"""
            intents:
            - {GREET_INTENT_NAME}
            - {GOODBYE_INTENT_NAME}
            actions:
            - {UTTER_GREET_ACTION}
            - {UTTER_ACTION_1}
            - {UTTER_ACTION_2}
            - {UTTER_ACTION_3}
            - {UTTER_ACTION_4}
            - {UTTER_ACTION_5}
            - {UTTER_BYE_ACTION}
            """
        )
        training_story = TrackerWithCachedStates.from_events(
            "training story",
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_ACTION_1),
                ActionExecuted(UTTER_ACTION_2),
                ActionExecuted(UTTER_ACTION_3),
                ActionExecuted(UTTER_ACTION_4),
                ActionExecuted(UTTER_ACTION_5),
                ActionExecuted(UTTER_BYE_ACTION),
                ActionExecuted(ACTION_LISTEN_NAME),
            ],
            domain=domain,
            slots=domain.slots,
        )
        interpreter = RegexInterpreter()
        policy.train([training_story], domain, interpreter)

        test_story1 = TrackerWithCachedStates.from_events(
            "test story",
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GOODBYE_INTENT_NAME}),
                ActionExecuted(UTTER_BYE_ACTION),
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_ACTION_1),
                ActionExecuted(UTTER_ACTION_2),
                ActionExecuted(UTTER_ACTION_3),
                ActionExecuted(UTTER_ACTION_4),
                ActionExecuted(UTTER_ACTION_5),
                # ActionExecuted(UTTER_BYE_ACTION),
            ],
            domain=domain,
            slots=domain.slots,
        )
        prediction1 = policy.predict_action_probabilities(
            test_story1, domain, interpreter
        )
        assert (
            domain.action_names_or_texts[
                prediction1.probabilities.index(max(prediction1.probabilities))
            ]
            == UTTER_BYE_ACTION
        )

        test_story2_no_match_expected = TrackerWithCachedStates.from_events(
            "test story",
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_BYE_ACTION),
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GOODBYE_INTENT_NAME}),
                ActionExecuted(UTTER_ACTION_1),
                ActionExecuted(UTTER_ACTION_2),
                ActionExecuted(UTTER_ACTION_3),
                ActionExecuted(UTTER_ACTION_4),
                ActionExecuted(UTTER_ACTION_5),
                # No prediction should be made here.
            ],
            domain=domain,
            slots=domain.slots,
        )

        prediction2 = policy.predict_action_probabilities(
            test_story2_no_match_expected, domain, interpreter
        )
        assert all([prob == 0.0 for prob in prediction2.probabilities])

    @pytest.mark.parametrize("max_history", [1, 2, 3, 4, None])
    def test_aug_pred_without_intent(self, max_history):
        """Tests memoization works for a memoized state sequence that does
        not have a user utterance.
        """
        policy = self.create_policy(
            featurizer=MaxHistoryTrackerFeaturizer(max_history=max_history), priority=1
        )

        GREET_INTENT_NAME = "greet"
        GOODBYE_INTENT_NAME = "goodbye"
        UTTER_GREET_ACTION = "utter_greet"
        UTTER_ACTION_1 = "utter_1"
        UTTER_ACTION_2 = "utter_2"
        UTTER_ACTION_3 = "utter_3"
        UTTER_ACTION_4 = "utter_4"
        domain = Domain.from_yaml(
            f"""
            intents:
            - {GREET_INTENT_NAME}
            - {GOODBYE_INTENT_NAME}
            actions:
            - {UTTER_GREET_ACTION}
            - {UTTER_ACTION_1}
            - {UTTER_ACTION_2}
            - {UTTER_ACTION_3}
            - {UTTER_ACTION_4}
            """
        )
        training_story = TrackerWithCachedStates.from_events(
            "training story",
            [
                ActionExecuted(UTTER_ACTION_3),
                ActionExecuted(UTTER_ACTION_4),
                ActionExecuted(ACTION_LISTEN_NAME),
            ],
            domain=domain,
            slots=domain.slots,
        )

        interpreter = RegexInterpreter()
        policy.train([training_story], domain, interpreter)

        test_story = TrackerWithCachedStates.from_events(
            "test story",
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_ACTION_1),
                ActionExecuted(UTTER_ACTION_2),
                ActionExecuted(UTTER_ACTION_3),
                # ActionExecuted(UTTER_ACTION_4),
            ],
            domain=domain,
            slots=domain.slots,
        )
        prediction = policy.predict_action_probabilities(
            test_story, domain, interpreter
        )
        assert (
            domain.action_names_or_texts[
                prediction.probabilities.index(max(prediction.probabilities))
            ]
            == UTTER_ACTION_4
        )
