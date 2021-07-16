import logging
from typing import Callable, Generator, Tuple, List, Optional, Dict, Text, TypeVar, Any


from rasa.shared.core.domain import State, SubState
from rasa.shared.core import state as state_utils
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import TEXT


logger = logging.getLogger(__name__)


class StateSequenceExtractor:
    """Extracts the right state sequence from the tracker.

    NOTE: if this makes things easier, we can also iterate over the "events" from the
    tracker here and apply filters on events instead of states
    """

    def __init__(
        self,
        omit_unset_slots: bool = True,
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
            if state_utils.is_prev_action_listen_in_state(
                state
            )  # FIXME: this only gives us answers to 'action_listen'
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


class PartialStateExtractor:
    """Extracts the information from a single state that should be featurized."""

    def __init__(
        self,
        input_schema: Dict[Text, List[Text]],
        input_schema_filter: Callable[[Dict[Text, List[Text]]], Dict[Text, List[Text]]],
        output_schema: Dict[Text, List[Text]],
    ):
        """
        Args:
          input_schema: the key/subkey combinations you'll want to extract from a state
            and featurize to obtain some input for a policy
          input_schema_filter: can be used in case certain key/subkey combinations
            should only be used iff certain other other key/subkey combinations that are
            (not) available (the input_schema is not sufficient to specify that);

        Example:
           If we *do not* want to use the TEXT attribute in case the INTENT attribute
           is populated, define an input_schema that covers TEXT *and* INTENT and then
           define an input_schema_filter that helps to map {USER:[TEXT,INTENT]} to
           {USER:[INTENT]}.
        """
        self.input_schema = input_schema
        self.input_schema_filter = input_schema_filter
        self.output_schema = set(output_schema)

    def extract_input(self, state: State) -> Dict[Text, SubState]:
        """

        """
        # yeah, this should not be implemented like this - but just to describe
        # what this should do:
        # (1) first filter all input_attributes
        input = state_utils.copy(state=state, key_dict=self.input_attributes)
        # (2) determine what is left and whether you still want that...
        input_schema = state_utils.schema(input)
        input_schema = self.input_spec_filter(input_schema)
        # (3) filter what you really wanted
        input = state_utils.copy(input=input, key_dict=input_schema)
        return input

    def extract_output(self, state: State) -> Dict[Text, SubState]:
        return state_utils.copy(state=state, key_dict=self.output_attributes)


T = TypeVar("T")


def unroll(
    items: List[T], min_window_end: int = 2, max_window_size: Optional[int] = None
) -> Generator[List[T]]:
    max_window_size = max_window_size if max_window_size else len(items)
    for rolling_end in range(min_window_end, len(items)):
        rolling_start = max(rolling_end - max_window_size, 0)
        yield items[rolling_start:rolling_end]


class MaxHistoryGenerator:
    """Helps to generate subsequences from state sequences.

    NOTE:  we could either
    - generate sub-sequences from sequences of states first -> then we have to featurize
      the same states again and again
    - generate sub-sequences from the featurized input/output tuples -> then the only
      computations we throw away are the "outputs" computed for the first max_history-1
      steps of a sequence (iff we don't catch that at the right point :) - that's not
      defined anywhere yet)
    """

    def __init__(self, max_window_size: Optional[int] = None,) -> None:

        self.max_window_size = max_window_size
        self._hashes = set()
        # TODO: This should NOT be persisted and loaded again, because it won't mean a
        # thing in the next session...

    def clear_history(self):
        self._hashes = set()

    def unroll_and_deduplicate(
        self, inputs_and_targets: List[Tuple[State, State]]
    ):  # FIXME: these should be partial states
        for subsequence in self.unroll_up_to(inputs_and_targets, self.max_history):
            input_states = [
                tuple[0] for tuple in subsequence[:-1]
            ]  # FIXME: [:-1] not in tracker featurizers....??
            last_target = subsequence[-1][1]
            hashed_subsequence = hash(
                tuple(
                    (
                        state_utils.FrozenState(s),
                        state_utils.FrozenState(last_target),
                    )  # FIXME: this works for partial states...
                    for s in input_states
                )
            )
            if hashed_subsequence not in self._hashes:
                yield subsequence
