from __future__ import annotations
from typing import List, Optional, Dict, Text, Any, ClassVar
from dataclasses import dataclass

from rasa.core.turns.turn import Turn, TurnParser
from rasa.shared.core.domain import Domain, State
from rasa.shared.core.events import ActionExecuted, Event, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    PREVIOUS_ACTION,
)
from rasa.shared.nlu.constants import ACTION_NAME


@dataclass
class ExtendedState(Turn):
    """A wrapper for the classic 'State'."""

    state: State
    events: Optional[List[Event]] = None

    BOT: ClassVar[str] = "bot"
    USER: ClassVar[str] = "user"

    def __post_init__(self):
        empty = [
            sub_state_name
            for sub_state_name, sub_state in self.state.items()
            if not sub_state
        ]
        for sub_state_name in empty:
            del self.state[sub_state_name]

    def __repr__(self) -> Text:
        optional = f":{self.events}" if self.events is not None else ""
        return f"{self.__class__.__name__}({self.get_type()}){optional}:{self.state}"

    def __str__(self) -> Text:
        return f"{self.__class__.__name__}({self.get_type()}):{self.state}"

    def get_type(self) -> str:
        return (
            self.USER
            if self.state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
            == ACTION_LISTEN_NAME
            else self.BOT
        )


@dataclass
class ExtendedStateParser(TurnParser):
    """
    Args:
        omit_unset_slots: If `True` do not include the initial values of slots.
        ignore_rule_only_turns: If set to `True`, we ignore `ActionExecuted` events
           that are the result of the application of a rule and hence we also do
           not create the corresponding turn.
        rule_only_data: Slots and loops, which only occur in rules but not in
          stories.
    """

    omit_unset_slots: bool = (False,)
    ignore_rule_only_turns: bool = (False,)
    rule_only_data: Optional[Dict[Text, Any]] = (None,)

    def parse(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
    ):
        """Returns all `RuleStates` created from the trackers event history.

        NOTE: By using the trackers `past_states` we to avoid the re-computation of
        cached states when given a `TrackerWithCachedStates` ...

        Args:
            tracker:
            domain: necessary evil needed until domain and state separated...
        Returns:
            ...
        """
        past_states = tracker.past_states(
            domain=domain,
            omit_unset_slots=self.omit_unset_slots,
            ignore_rule_only_turns=self.ignore_rule_only_turns,
            rule_only_data=self.rule_only_data,
        )
        return [ExtendedState(state=state) for state in past_states]

    def parse_with_events(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> List[ExtendedState]:
        """Returns all `RuleStates` from the trackers event history.
        NOTE: This could be a replacement for `domain.states_for_tracker_history` and
        `tracker.past_states` - if we didn't use the `TrackerWithCachedStates` which
        pre-computes and caches states during data loading.
        Args:
            tracker: Dialogue state tracker containing the dialogue so far.
            domain: the common domain
        Return:
            ...
        """
        turns: List[ExtendedState] = [
            ExtendedState(
                state={},
                events=[],
            )
        ]

        replay_tracker = tracker.init_copy()
        # FIXME: Which settings are copied in this call that are crucial to get the
        #  correct results? Can we make this function take only (applied) events as
        #  input?
        state_actor = self.BOT
        state_events = []

        events = tracker.applied_events()

        # if ignore_rule_only_turns:
        last_turn_was_hidden = False
        last_non_hidden_action_executed = None

        for idx, current_event in enumerate(events):

            # Apply the event
            replay_tracker.update(current_event)
            state_events.append(current_event)

            # Update actor information
            if isinstance(current_event, UserUttered):
                if not self.ignore_rule_only_turns and state_actor == self.USER:
                    raise RuntimeError(
                        "Found a `UserUttered` event that was possibly followed "
                        "by some events that were *not* `ActionExecutedEvents` "
                        "before encountering the next `UserUttered` event, "
                        "event though we are not ignoring rule turns."
                    )
                state_actor = cls.USER

            # Generate a turn ...
            if idx == len(events) - 1 or isinstance(events[idx + 1], ActionExecuted):

                # At this point the **tracker** has seen all the events that will be
                # used to produce the next state, i.e.
                #         `events[0], ..., events[idx] = current_event`
                # (ignoring hidden events).
                #
                # Let's assume j is the index such that
                #         `events[j], ..., events[idx] = current_event`
                # contribute to this next state but not to the previous one, then
                #         `events[j], ..., events[idx], events[idx+1]`
                # is what we call the **current turn**.
                # Observe that:
                # - `events[idx+1]` is an `ActionExecuted` and
                # - `events[j],...,events[idx-1]` are *not* `ActionExecuted` events
                #
                # Let's denote the events of the **previous turn** by
                #         `events[h], ..., events[j-1]`
                # Note that `events[j-1]` is an `ActionExecuted`.

                if self.ignore_rule_only_turns:

                    # (1) Memorize the last non hidden action
                    # If the *last* turn was not hidden, then we should remember
                    # it's non-hidden `ActionExecuted` because the current turn
                    # might be hidden and hence we'd ignore it's `ActionExecuted`
                    # as `previous_action` information.
                    # By design, this `ActionExecuted` event is `events[j-1]` which
                    # is the last last `ActionExecuted` event seen by the tracker.
                    if not last_turn_was_hidden:
                        last_non_hidden_action_executed = replay_tracker.latest_action

                    # (2) Should this turn be hidden?

                    # (2.a) Not if it is no proper turn...
                    # This turn can't be hidden if it is no proper turn, i.e.
                    # there is no `event[idx+1]`.
                    if idx < len(events) - 1:

                        # (2.b) Yes, if it ends with a `hide_rule_turn` event, but not
                        #       in some special case. Also we may hide some
                        #       `ActionExecuted` events instead of whole turns.
                        # If the last turn ended with a follow up action or an
                        # active loop prediction, then
                        # - this turn will **not** be hidden *and*
                        # - in the next round, we will act like this turn **was**
                        #   hidden iff the turn prior to this turn was hidden
                        #   regarding the `last_non_hidden_action_executed` (only!)
                        # (As described above, last `ActionExecuted` seen by
                        # the tracker is `events[j-1]`. Hence, the following translates
                        # to the previous turn ending in a specific event.)
                        if not replay_tracker.followup_action and (
                            not replay_tracker.latest_action_name
                            == replay_tracker.active_loop_name
                        ):
                            next_event = events[idx + 1]
                            last_turn_was_hidden = next_event.hide_rule_turn
                            if last_turn_was_hidden:
                                continue

                # Create a state using `events[0], ..., events[idx] = current_event`
                # (ignoring hidden events).
                turn_state = domain.get_active_state(
                    replay_tracker, omit_unset_slots=self.omit_unset_slots
                )

                if self.ignore_rule_only_turns:
                    # Clean *every* state from only rule features
                    domain._remove_rule_only_features(turn_state, self.rule_only_data)

                    # The state that we just created does not capture an `ActionListen`,
                    # then:
                    # - If the previous state did not contain a user sub-state, then
                    #   this state also cannot contain a user sub-state (and it will
                    #   be removed)
                    # - If the previous state did contain a user sub-state, then
                    #   this state must contain the same user sub-state.
                    # This is necessary because # FIXME: Why is this needed?
                    domain._substitute_rule_only_user_input(turn_state, turns[-1].state)

                    # The replay tracker may have seen `ActionExecuted` events that
                    # we actually want to ignore due to (2.b)
                    if last_non_hidden_action_executed:
                        turn_state[PREVIOUS_ACTION] = last_non_hidden_action_executed

                    # Remove empty sub-states! (TODO: where is this needed downstream?)
                    self.remove_empty_sub_states(turn_state)

                # Note: If the turn just captures an `ActionListen`. In this
                # case, we attribute the turn to the user, even though the turn does
                # not capture a new utterance to handle `ActionListen` consistently.

                # Finish the turn
                turn = ExtendedState(state=turn_state, events=state_events)
                turns.append(turn)

                # Reset
                state_events = []

        return turns
