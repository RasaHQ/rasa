from abc import abstractmethod, ABC
import enum
from pathlib import Path
from dataclasses import dataclass
import dataclasses
from rasa.core import featurizers

import jsonpickle
import logging

from tqdm import tqdm
from typing import Generator, Tuple, List, Optional, Dict, Text, Any
import numpy as np

from rasa.shared.core.domain import State
from rasa.shared.core import state as state_utils
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import TEXT, ENTITIES
from rasa.shared.exceptions import RasaException

FEATURIZER_FILE = "featurizer.json"  # FIXME: persist/load

logger = logging.getLogger(__name__)


def get_entity_data(event: UserUttered) -> Dict[Text, Any]:
    # TODO: this should go into UserUttered and return MessageData ...

    if event.text and not event.intent_name:
        return {TEXT: event.text, ENTITIES: event.entities}

    # input is not textual, so add empty dict
    return {}


class InvalidStory(RasaException):
    """Exception that can be raised if story cannot be featurized."""

    def __init__(self, message: Text) -> None:
        self.message = message
        super(InvalidStory, self).__init__()

    def __str__(self) -> Text:
        return self.message


@dataclass
class Targets:
    """
    FIXME:TODO: this should be replaced by state, state-like or a sub-state class...

    """

    intent: Optional[Text] = None
    actions: Optional[Text] = None
    entities: Optional[List[Text]] = None


class Extractor:
    """ TODO: this should be part of Tracker... ?"""

    def extract_for_training(
        tracker: DialogueStateTracker, omit_unset_slots: bool = False,
    ) -> Tuple[List[State], Targets]:
        """Transforms a tracker into a lists of states and targets (e.g. actions).
        Args:
            trackers: The trackers to transform
            omit_unset_slots: If `True` do not include the initial values of slots
              in the extracted states.
        Returns:
            A tuple of a list of states and a list of target objects
        """

        # NOTE: we *must not* start to ignore rule only turns here or we'll loose the
        # alignment of events and states
        all_past_states = tracker.past_states(omit_unset_slots=omit_unset_slots)
        all_past_events = tracker.applied_events()

        assert len(all_past_states) == len(all_past_events)

        states = []
        actions = []
        entities = []

        for state, event in zip(all_past_states, all_past_events):

            if not isinstance(event, ActionExecuted) or event.unpredictable:
                continue

            entity_data = {}
            if isinstance(event, UserUttered):
                entity_data = get_entity_data(event)

            if state_utils.get_intent(state):
                state_utils.forget_user_text(state)

            states.append(state)
            actions.append(event.action_name or event.action_text)
            entities.append(entity_data)

        return (
            state,
            Targets(actions=actions, entities=entities),
        )

    def extract_for_prediction(
        self,
        tracker: DialogueStateTracker,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        ignored_active_loop_names: Optional[List[Text]] = None,
        ignored_slots: Optional[List[Text]] = None,
    ) -> List[State]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform.
            domain: The domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: Indicates whether we should ignore dialogue turns
               that are present only in rules.
               **You will want to set this to true iff you have applied
               `extract_for_training` to data that did not contain RuleSteps**.
            ignored_active_loop_names: active loops whose names appear in this list will
              be ignored if and only if `ignore_rule_only_turns` is True
            ignored_slots: slots in this list will be ignored if and only if
               `ignore_rule_only_turns` is True
        Returns:
            A list of states.
        """
        states = tracker.past_states(
            ignore_rule_only_turns=ignore_rule_only_turns,
            ignored_active_loop_names=ignored_active_loop_names,
            ignored_slots=ignored_slots,
        )
        state_utils.forget_states_after_last_user_input(
            states, use_text_for_last_user_input
        )
        return states


class UnRoller:
    """
    TODO: there is probably a better name for this.... :)
    """

    def __init__(
        self, max_history: Optional[int] = None, remove_duplicates: bool = True,
    ) -> None:

        self.max_history = max_history
        # from multiple states that create equal featurizations
        # we only need to keep one.
        self._hashes = set()

    def clear_history(self):
        self._hashes = set()

    @staticmethod
    def _hash_example(
        states: List[State], action: Text, tracker: DialogueStateTracker
    ) -> int:
        """Hash states for efficient deduplication."""
        frozen_states = tuple(
            s if s is None else tracker.freeze_current_state(s) for s in states
        )
        frozen_actions = (action,)
        return hash((frozen_states, frozen_actions))

    def unroll_and_deduplicate(self, states_and_targets: List[Tuple[State, Targets]]):
        """
        FIXME
        """
        for subsequence in self.unroll_up_to(states_and_targets, self.max_history):
            last_target = subsequence[-1][1]
            hash = self._hash_example(
                [tup[0] for tup in subsequence],
                last_target.action
                or last_target.intent,  # TODO: not so nice (maybe we'll have both)
            )
            if hash not in self._hashes:
                yield subsequence

    # FIXME
    import typing

    T = typing.TypeVar("T")

    @staticmethod
    def unroll(items: List[T], max_history: Optional[int] = None) -> Generator[List[T]]:
        """
        FIXME: test :)
        """
        window = max_history if max_history is not None else len(items)
        for rolling_end in range(1, len(items)):
            rolling_start = max(rolling_end - window, 0)
            yield items[rolling_start:rolling_end]

    @staticmethod
    def unroll_for_prediction(states: List[State], max_history: Optional[int] = None):
        """
        TODO: remove this?

        Note: no need to _choose_last_user_input because Extractor has taken care of that
        """
        return [
            state_sequence
            for state_sequence in UnRoller.unroll(items=states, max_history=max_history)
        ]
