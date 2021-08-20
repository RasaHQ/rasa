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
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            SlotSet("slot_1", True),
            ActionExecuted(UTTER_GREET_ACTION),
            SlotSet("slot_2", True),
            SlotSet("slot_3", True),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(UTTER_GREET_ACTION),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            SlotSet("slot_4", True),
            ActionExecuted(UTTER_BYE_ACTION),
        ]
        training_story = TrackerWithCachedStates.from_events(
            "training story", evts=events, domain=domain, slots=domain.slots,
        )
        test_story = TrackerWithCachedStates.from_events(
            "training story", events[:-1], domain=domain, slots=domain.slots,
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
                ActionExecuted(UTTER_GREET_ACTION),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_3", True),
                ActionExecuted(UTTER_BYE_ACTION),
            ],
            domain=domain,
            slots=domain.slots,
        )
        test_story = TrackerWithCachedStates.from_events(
            "test story",
            [
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_1", False),
                ActionExecuted(UTTER_GREET_ACTION),
                ActionExecuted(UTTER_GREET_ACTION),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_2", True),
                ActionExecuted(UTTER_GREET_ACTION),
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
