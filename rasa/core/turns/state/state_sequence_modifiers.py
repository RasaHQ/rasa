from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Any
import copy

from rasa.core.turns.state.state import ExtendedState
from rasa.core.turns.to_dataset.turn_sub_sequence_generator import TurnSequenceModifier
from rasa.shared.core.constants import (
    ACTION_UNLIKELY_INTENT_NAME,
    PREVIOUS_ACTION,
    USER,
)
from rasa.shared.nlu.constants import ACTION_NAME, ENTITIES, INTENT, TEXT


class RemoveLastStateIfUserState(TurnSequenceModifier[ExtendedState]):
    def modify(
        self,
        turns: List[ExtendedState],
        training: bool,
        inplace_allowed: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtendedState]:
        if turns and turns[-1].get_type() == ExtendedState.USER:
            return turns[:-1]
        return turns


class RemoveStatesWithPrevActionUnlikelyIntent(TurnSequenceModifier[ExtendedState]):
    """Remove turns where the previous action substate is an action unlikely intent."""

    def modify(
        self,
        turns: List[ExtendedState],
        training: bool,
        inplace_allowed: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtendedState]:
        return [
            turn
            for turn in turns
            if turn.state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
            != ACTION_UNLIKELY_INTENT_NAME
        ]


@dataclass
class IfLastStateWasUserStateKeepEitherTextOrNonText(
    TurnSequenceModifier[ExtendedState]
):
    """Removes (intent and entities) or text from the last turn if it was a user turn.

    Only does so during inference. During training, nothing is removed.

    TODO: why was this always applied - shouldn't it only be done if intent/entities
    are targets? (in that case, the new label extractors would take care of this)
    """

    keep_text: Callable[
        [List[ExtendedState], Optional[Dict[str, Any]], bool], bool
    ] = lambda *args: True

    def modify(
        self,
        turns: List[ExtendedState],
        training: bool,
        inplace_allowed: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtendedState]:
        if turns and turns[-1].get_type() == ExtendedState.USER:
            last_turn = turns[-1]
            if last_turn.state.get(USER) and last_turn.state.get(INTENT):
                if not inplace_allowed:
                    last_turn = copy.deepcopy(last_turn)
                remove = (
                    [INTENT, ENTITIES]
                    if self.keep_text(turns, context, training)
                    else [TEXT]
                )
                for key in remove:
                    last_turn.state.get(USER, {}).pop(key, None)

            turns[-1] = last_turn
        return turns


class RemoveUserTextIfIntentFromEveryState(TurnSequenceModifier[ExtendedState]):
    """Removes the text if there is an intent in the user substate of every turn.

    This is always applied - during training as well as during inference.
    """

    def modify(
        self,
        turns: List[ExtendedState],
        training: bool,
        inplace_allowed: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtendedState]:

        for idx in range(len(turns)):
            turn = turns[idx]
            if not inplace_allowed:
                turn = copy.deepcopy(turn)
            user_sub_state = turn.state.get(USER, {})
            if TEXT in user_sub_state and INTENT in user_sub_state:
                del user_sub_state[TEXT]
            if inplace_allowed:
                turns[idx] = turn
        return turns
