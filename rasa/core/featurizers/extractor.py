import logging
from typing import Callable, Generator, Tuple, List, Optional, Dict, Text, Any
import typing


from rasa.shared.core.domain import State
from rasa.shared.core import state as state_utils
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import TEXT

FEATURIZER_FILE = "featurizer.json"  # FIXME: persist/load

logger = logging.getLogger(__name__)


class Extractor:
    def __init__(
        self,
        input_attributes: Dict[Text, List[Text]],
        output_attributes: Dict[Text, List[Text]],
        state_filter: Optional[Callable[[State], bool]] = True,
    ):
        self.input_attributes = set(input_attributes)
        self.ouput_state_attributes = set(output_attributes)
        self.state_filter = state_filter

    def extract_for_training(
        self, tracker: DialogueStateTracker, omit_unset_slots: bool = False,
    ) -> List[Tuple[State, State]]:

        input_output_sequences = []
        for state in enumerate(tracker.past_states(omit_unset_slots=omit_unset_slots)):

            # skip the state if the filter says so - or if it isn't a user
            # utterance
            if (self.state_filter and self.state_filter(state)) or (
                not state_utils.is_prev_action_listen_in_state(state)
            ):
                continue

            # NOTE: Why don't we skip the first state which is an "unpredictable"
            # event? Because it's still a valid input. We'll take care of not asking
            # to predict from empty input later.

            input_state = state_utils.copy(state=state, key_dict=self.input_attributes)
            # remove text if an intent is given to avoid working on intent level
            if state_utils.get_user_intent(input_state):
                state_utils.forget_user_text(input_state)

            output_state = state_utils.copy(
                state=state, key_dict=self.output_attributes
            )

            input_output_sequences.append((input_state, output_state))

        return input_output_sequences

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


T = typing.TypeVar("T")


def unroll(items: List[T], max_history: Optional[int] = None) -> Generator[List[T]]:
    """
    FIXME:
    """
    window = max_history if max_history is not None else len(items)
    # Note that we start with "2" so that we have at least one input state.
    for rolling_end in range(2, len(items)):
        rolling_start = max(rolling_end - window, 0)
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

        Note: no need to _choose_last_user_input because Extractor has taken care of that
        """
        return [
            state_sequence
            for state_sequence in UnRoller.unroll(items=states, max_history=max_history)
        ]
