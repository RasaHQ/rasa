
from rasa.shared.core.trackers import DialogueStateTracker
from typing import TypedDict, Text, List, Optional, Any, Set, Union, Tuple
from rasa.shared.core.constants import ACTION_LISTEN_NAME, ACTION_UNLIKELY_INTENT_NAME
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    Event,
    EntitiesAdded,
    ActiveLoop,
)
from rasa.shared.core.domain import Domain
from collections import defaultdict


class DialogueTurnState:
    """
    'slots': {'abc': (1.0, 1.0)}
    'user': {'text': "I'm sad"}
    'user': {'intent': 'greet'}
    'prev_action': {'action_name': 'utter_did_that_help'}
    'prev_action': {'action_text': 'too bad'}

    we add:

    'flags': ['is_rule_only_turn', 'is_augmented']

    """
    def __init__(self):
        self.features: Dict[Text, Dict] = defaultdict(dict)
        self.flags: Set[Text] = set()

    def add_feature(
        self,
        group: Text,
        name: Text,
        value: Union[Text, Tuple[float]]
    ) -> None:
        self.features[group][name] = value

    def flag(self, flag_name: Text) -> None:
        self.flags.add(flag_name)

    def __repr__(self) -> Text:
        return (
            f"{self.__class__.__name__}({dict(self.features)}, {self.flags})"
        )


class FeatureSpace:

    def __init__(
        self,
        feature_space_description: Dict[Text, Text],
    ):
        self._feature_space_description = feature_space_description
        self._feature_names: Dict[Text, Dict[Text, Set[int]]] = {
            group: defaultdict(set)
            for group in self.feature_space_description
        }

    def expand(self, turn: DialogueTurnState) -> None:
        for group, name in self._feature_space_description.items():
            if group in turn.features and name in turn.features[group]:
                self._feature_names[group][name].add(hash(turn.features[group][name]))


    def add_subspace(
        self,
        group: Text,
        name: Text,
        size: int = 0,
    ) -> None:
        self.feature_space[group][name] = size


class DialogueState:

    def __init__(
        self,
        turn_states: List[DialogueTurnState] = {},
        origin: Optional[Text] = None
    ):
        self.origin = origin
        self._turn_states = turn_states

    def __getitem__(self, index: int) -> DialogueTurnState:
        return self._turn_states[index]

    def __len__(self) -> int:
        return len(self._turn_states)


class Policy:

    def __init__(self, name: Text):
        self.name = name
        self.input_space = FeatureSpace(
            {
                ("action", "action_name"): ONE_HOT_ENCODER,
                ("action", "action_text"): ONE_HOT_ENCODER,
                ("user", "intent"): ONE_HOT_ENCODER,
                ("user", "text"): self.user_text_featurizer,
                "slot": PRE_ENCODED,
            }
        )
        self.output_space = FeatureSpace(
            {
                ("action", "action_name"): ONE_HOT_ENCODER,
                ("action", "action_text"): ONE_HOT_ENCODER,
                ("entity", ""): 0,
                # ("user", "intent"): MULTI_HOT_ENCODER,
            }
        )

    def train(
        self,
        dialogues: List[DialogueState],
        domain: Optional[Domain] = None,
    ):
        # filter dialogues
        # filter turn states
        # build "narrow domain"
        # featurize turn states
        filtered_dialogues = self.preprocess_for_training(dialogues)
        states, labels = self.training_states_and_labels(filtered_dialogues)

    def preprocess_for_training(dialogues: List[DialogueState]) -> List[DialogueState]:
        for dialogue in dialogues:
            for turn in dialogue:
                self.input_space.expand(trun)
                self.output_space.expand(trun)
        return dialogues

    def predict_action(
        self,
        dialogue_state: DialogueState,
    ):
        pass


class PolicyEnsemble:

    def __init__(self, policies: List[Policy]):
        self.polices = policies

    def train(self, trackers: List[DialogueStateTracker], domain: Domain):
        trackers_as_turn_states = convert_trackers_to_turn_states(trackers, domain)
        for s in trackers_as_turn_states[0]:
            print(s)
        for policy in self.polices:
            # ToDo: copy or reset trackers_as_turn_states
            policy.train(trackers_as_turn_states, domain)


def convert_trackers_to_turn_states(
    trackers: List[DialogueStateTracker],
    domain: Domain,
) -> List[DialogueState]:
    return [
        _convert_tracker_to_turn_states(tracker, domain)
        for tracker in trackers
    ]  # use yield instead


def _convert_tracker_to_turn_states(
    tracker: DialogueStateTracker,
    domain: Domain,
) -> DialogueState:
    origin = "rule_tracker" if tracker.is_rule_tracker else "story_tracker"
    states = []
    for tr, hide_rule_turn in tracker.generate_all_prior_trackers():
        flags = []
        if hide_rule_turn:
            flags.append("hide_rule_turn")
        states.append(_get_tracker_turn_state(tr, domain, flags))

    return DialogueState(states, origin)


def _get_tracker_turn_state(
    tracker: DialogueStateTracker,
    domain: Domain,
    flags: List[Text] = [],
) -> DialogueTurnState:
    state = DialogueTurnState()
    for flag in flags:
        state.flag(flag)

    # Add slots
    for slot_name, slot in tracker.slots.items():
        if slot is not None and slot.as_feature():
            if slot.value == rasa.shared.core.constants.SHOULD_NOT_BE_SET:
                state.add_feature("slots", slot_name, rasa.shared.core.constants.SHOULD_NOT_BE_SET)
            elif any(slot.as_feature()):
                # only add slot if some of the features are not zero
                state.add_feature("slots", slot_name, tuple(slot.as_feature()))

    # Add action
    for key, value in tracker.latest_action.items():
        state.add_feature("action", key, value)

    # Add user
    latest_message = tracker.latest_message
    if latest_message and not latest_message.is_empty():
        # print(latest_message.as_sub_state())
        for key, value in latest_message.as_sub_state().items():
            state.add_feature("user", key, value)
            # ToDo: deal with entities

    return state

if __name__ == "__main__":
    UTTER_GREET_ACTION = "utter_greet"
    GREET_INTENT_NAME = "greet"
    DOMAIN_YAML = f"""
    intents:
    - {GREET_INTENT_NAME}
    actions:
    - {UTTER_GREET_ACTION}
    """
    domain = Domain.from_yaml(DOMAIN_YAML)
    training_trackers = [
        DialogueStateTracker.from_events("sender_id", [
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION, hide_rule_turn=True),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(ACTION_LISTEN_NAME),
        ])
    ]
    # print(training_trackers[0].events)
    policy1 = Policy("rule")
    policy2 = Policy("ted")
    ensemble = PolicyEnsemble([policy1, policy2])

    ensemble.train(training_trackers, domain)

    # test_tracker = DialogueStateTracker.from_events("sender_id", [
    #     # ...
    # ])
    # prediction = ensemble.predict_action(test_tracker)
