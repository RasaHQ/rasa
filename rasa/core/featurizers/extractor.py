import logging
from typing import Callable, Generator, Tuple, List, Optional, Dict, Text, TypeVar, Any


from rasa.shared.core.domain import State
from rasa.shared.core import state as state_utils
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import TEXT


logger = logging.getLogger(__name__)


class StateFilter:
    """TODO: move this to tracker?"""

    def __init__(
        self,
        omit_unset_slots: bool = True,  # FIXME: this should not be here... -> MessageDataExtractor?
        extra_state_filter: Optional[Callable[[State], bool]] = True,
    ):
        self.extra_state_filter = extra_state_filter
        self.omit_unset_slots = omit_unset_slots

    def extract_for_training(
        self, tracker: DialogueStateTracker, omit_unset_slots: bool = False,
    ) -> List[State]:

        return [
            state
            for state in tracker.past_states(omit_unset_slots=omit_unset_slots)
            if state_utils.is_prev_action_listen_in_state(state)
            and not self.extra_state_filter(state)
        ]

    @staticmethod
    def extract_for_prediction(
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
            states, use_text_for_last_user_input=use_text_for_last_user_input,
        )
        return states


class MessageDataExtractor:
    """Converts a State into input or output message data that can be featurized."""

    def __init__(
        self,
        input_attributes: Dict[Text, List[Text]],
        output_attributes: Dict[Text, List[Text]],
    ):
        self.input_attributes = set(input_attributes)
        self.ouput_state_attributes = set(output_attributes)

    def extract_input(self, state: State) -> Dict[Text, Any]:
        state = state_utils.copy(state=state, key_dict=self.input_attributes)
        # remove text if an intent is given to avoid working on intent level
        if state_utils.get_user_intent(state):
            state_utils.forget_user_text(state)
        return state

    def extract_output(self, state: State) -> Dict[Text, Any]:
        return state_utils.copy(state=state, key_dict=self.output_attributes)


T = TypeVar("T")


def unroll(
    items: List[T], min_window_end: int = 2, max_window_size: Optional[int] = None
) -> Generator[List[T]]:
    max_window_size = max_window_size if max_window_size else len(items)
    for rolling_end in range(min_window_end, len(items)):
        rolling_start = max(rolling_end - max_window_size, 0)
        yield items[rolling_start:rolling_end]


class UnRoller:
    """
    TODO: there is probably a better name for this.... :)
    FIXME: state and entities/actions tuples always belong to the same step
     --> need to cutoff the last state...
    """

    def __init__(self, max_history: Optional[int] = None,) -> None:

        self.max_history = max_history
        # from multiple states that create equal featurizations
        # we only need to keep one.
        self._hashes = set()
        # TODO: This should NOT be persisted and loaded again, because it won't mean a
        # thing in the next session...

    def clear_history(self):
        self._hashes = set()

    def unroll_and_deduplicate(self, inputs_and_targets: List[Tuple[State, State]]):
        for subsequence in self.unroll_up_to(inputs_and_targets, self.max_history):
            input_states = [
                tuple[0] for tuple in subsequence[:-1]
            ]  # FIXME: [:-1] not in tracker featurizers....??
            last_target = subsequence[-1][1]
            hashed_subsequence = hash(
                tuple(
                    (state_utils.FrozenState(s), state_utils.FrozenState(last_target))
                    for s in input_states
                )
            )
            if hashed_subsequence not in self._hashes:
                yield subsequence

    @staticmethod
    def unroll_for_prediction(states: List[State], max_history: Optional[int] = None):
        """
        TODO: remove this?

        NOTE: no need to _choose_last_user_input because Extractor has taken care of that
        """
        return [
            state_sequence
            for state_sequence in UnRoller.unroll(items=states, max_history=max_history)
        ]
