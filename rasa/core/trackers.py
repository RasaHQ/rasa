import copy
import logging
from collections import deque
from enum import Enum
from typing import (
    Dict,
    Text,
    Any,
    Optional,
    Iterator,
    Generator,
    Type,
    List,
    Deque,
    Iterable,
    Union,
)

import typing

from rasa.nlu.constants import (
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_GROUP,
)
from rasa.core import events  # pytype: disable=pyi-error
from rasa.core.actions.action import ACTION_LISTEN_NAME  # pytype: disable=pyi-error
from rasa.core.conversation import Dialogue  # pytype: disable=pyi-error
from rasa.core.events import (  # pytype: disable=pyi-error
    UserUttered,
    ActionExecuted,
    Event,
    SlotSet,
    Restarted,
    ActionReverted,
    UserUtteranceReverted,
    BotUttered,
    ActiveLoop,
    SessionStarted,
    ActionExecutionRejected,
)
from rasa.core.domain import Domain  # pytype: disable=pyi-error
from rasa.core.slots import Slot
from rasa.utils import common as common_utils


if typing.TYPE_CHECKING:
    from rasa.core.training.structures import Story

logger = logging.getLogger(__name__)


ACTIVE_LOOP_KEY = "active_loop"


class EventVerbosity(Enum):
    """Filter on which events to include in tracker dumps."""

    # no events will be included
    NONE = 1

    # all events, that contribute to the trackers state are included
    # these are all you need to reconstruct the tracker state
    APPLIED = 2

    # include even more events, in this case everything that comes
    # after the most recent restart event. this will also include
    # utterances that got reverted and actions that got undone.
    AFTER_RESTART = 3

    # include every logged event
    ALL = 4


class AnySlotDict(dict):
    """A slot dictionary that pretends every slot exists, by creating slots on demand.

    This only uses the generic slot type! This means certain functionality wont work,
    e.g. properly featurizing the slot."""

    def __missing__(self, key) -> Slot:
        value = self[key] = Slot(key)
        return value

    def __contains__(self, key) -> bool:
        return True


class DialogueStateTracker:
    """Maintains the state of a conversation.

    The field max_event_history will only give you these last events,
    it can be set in the tracker_store"""

    @classmethod
    def from_dict(
        cls,
        sender_id: Text,
        events_as_dict: List[Dict[Text, Any]],
        slots: Optional[List[Slot]] = None,
        max_event_history: Optional[int] = None,
    ) -> "DialogueStateTracker":
        """Create a tracker from dump.

        The dump should be an array of dumped events. When restoring
        the tracker, these events will be replayed to recreate the state."""

        evts = events.deserialise_events(events_as_dict)
        return cls.from_events(sender_id, evts, slots, max_event_history)

    @classmethod
    def from_events(
        cls,
        sender_id: Text,
        evts: List[Event],
        slots: Optional[List[Slot]] = None,
        max_event_history: Optional[int] = None,
        sender_source: Optional[Text] = None,
    ):
        tracker = cls(sender_id, slots, max_event_history, sender_source)
        for e in evts:
            tracker.update(e)
        return tracker

    def __init__(
        self,
        sender_id: Text,
        slots: Optional[Iterable[Slot]],
        max_event_history: Optional[int] = None,
        sender_source: Optional[Text] = None,
        is_rule_tracker: bool = False,
    ) -> None:
        """Initialize the tracker.

        A set of events can be stored externally, and we will run through all
        of them to get the current state. The tracker will represent all the
        information we captured while processing messages of the dialogue."""

        # maximum number of events to store
        self._max_event_history = max_event_history
        # list of previously seen events
        self.events = self._create_events([])
        # id of the source of the messages
        self.sender_id = sender_id
        # slots that can be filled in this domain
        if slots is not None:
            self.slots = {slot.name: copy.copy(slot) for slot in slots}
        else:
            self.slots = AnySlotDict()
        # file source of the messages
        self.sender_source = sender_source
        # whether the tracker belongs to a rule-based data
        self.is_rule_tracker = is_rule_tracker

        ###
        # current state of the tracker - MUST be re-creatable by processing
        # all the events. This only defines the attributes, values are set in
        # `reset()`
        ###
        # if tracker is paused, no actions should be taken
        self._paused = False
        # A deterministically scheduled action to be executed next
        self.followup_action = ACTION_LISTEN_NAME
        self.latest_action_name = None
        # Stores the most recent message sent by the user
        self.latest_message = None
        self.latest_bot_utterance = None
        self._reset()
        self.active_loop: Dict[Text, Union[Text, bool, Dict, None]] = {}

    ###
    # Public tracker interface
    ###
    def current_state(
        self, event_verbosity: EventVerbosity = EventVerbosity.NONE
    ) -> Dict[Text, Any]:
        """Return the current tracker state as an object."""

        _events = self._events_for_verbosity(event_verbosity)
        if _events:
            _events = [e.as_dict() for e in _events]
        latest_event_time = None
        if len(self.events) > 0:
            latest_event_time = self.events[-1].timestamp

        return {
            "sender_id": self.sender_id,
            "slots": self.current_slot_values(),
            "latest_message": self.latest_message.parse_data,
            "latest_event_time": latest_event_time,
            "followup_action": self.followup_action,
            "paused": self.is_paused(),
            "events": _events,
            "latest_input_channel": self.get_latest_input_channel(),
            ACTIVE_LOOP_KEY: self.active_loop,
            "latest_action_name": self.latest_action_name,
        }

    def _events_for_verbosity(
        self, event_verbosity: EventVerbosity
    ) -> Optional[List[Event]]:
        if event_verbosity == EventVerbosity.ALL:
            return list(self.events)
        if event_verbosity == EventVerbosity.AFTER_RESTART:
            return self.events_after_latest_restart()
        if event_verbosity == EventVerbosity.APPLIED:
            return self.applied_events()

        return None

    def past_states(self, domain) -> deque:
        """Generate the past states of this tracker based on the history."""

        generated_states = domain.states_for_tracker_history(self)
        return deque(frozenset(s.items()) for s in generated_states)

    def change_loop_to(self, loop_name: Text) -> None:
        """Set the currently active loop.

        Args:
            loop_name: The name of loop which should be marked as active.
        """
        if loop_name is not None:
            self.active_loop = {
                "name": loop_name,
                "validate": True,
                "rejected": False,
                "trigger_message": self.latest_message.parse_data,
            }
        else:
            self.active_loop = {}

    def change_form_to(self, form_name: Text) -> None:
        common_utils.raise_warning(
            "`change_form_to` is deprecated and will be removed "
            "in future versions. Please use `change_loop_to` "
            "instead.",
            category=DeprecationWarning,
        )
        self.change_loop_to(form_name)

    def set_form_validation(self, validate: bool) -> None:
        """Toggle form validation"""
        self.active_loop["validate"] = validate

    def reject_action(self, action_name: Text) -> None:
        """Notify active loop that it was rejected"""
        if action_name == self.active_loop.get("name"):
            self.active_loop["rejected"] = True

    def set_latest_action_name(self, action_name: Text) -> None:
        """Set latest action name
            and reset form validation and rejection parameters
        """
        self.latest_action_name = action_name
        if self.active_loop.get("name"):
            # reset form validation if some loop is active
            self.active_loop["validate"] = True
        if action_name == self.active_loop.get("name"):
            # reset loop rejection if it was predicted again
            self.active_loop["rejected"] = False

    def current_slot_values(self) -> Dict[Text, Any]:
        """Return the currently set values of the slots"""
        return {key: slot.value for key, slot in self.slots.items()}

    def get_slot(self, key: Text) -> Optional[Any]:
        """Retrieves the value of a slot."""

        if key in self.slots:
            return self.slots[key].value
        else:
            logger.info(f"Tried to access non existent slot '{key}'")
            return None

    def get_latest_entity_values(
        self,
        entity_type: Text,
        entity_role: Optional[Text] = None,
        entity_group: Optional[Text] = None,
    ) -> Iterator[Text]:
        """Get entity values found for the passed entity type and optional role and
        group in latest message.

        If you are only interested in the first entity of a given type use
        `next(tracker.get_latest_entity_values("my_entity_name"), None)`.
        If no entity is found `None` is the default result.

        Args:
            entity_type: the entity type of interest
            entity_role: optional entity role of interest
            entity_group: optional entity group of interest

        Returns:
            Entity values.
        """

        return (
            x.get(ENTITY_ATTRIBUTE_VALUE)
            for x in self.latest_message.entities
            if x.get(ENTITY_ATTRIBUTE_TYPE) == entity_type
            and (entity_group is None or x.get(ENTITY_ATTRIBUTE_GROUP) == entity_group)
            and (entity_role is None or x.get(ENTITY_ATTRIBUTE_ROLE) == entity_role)
        )

    def get_latest_input_channel(self) -> Optional[Text]:
        """Get the name of the input_channel of the latest UserUttered event"""

        for e in reversed(self.events):
            if isinstance(e, UserUttered):
                return e.input_channel
        return None

    def is_paused(self) -> bool:
        """State whether the tracker is currently paused."""
        return self._paused

    def idx_after_latest_restart(self) -> int:
        """Return the idx of the most recent restart in the list of events.

        If the conversation has not been restarted, ``0`` is returned."""

        for i, event in enumerate(reversed(self.events)):
            if isinstance(event, Restarted):
                return len(self.events) - i

        return 0

    def events_after_latest_restart(self) -> List[Event]:
        """Return a list of events after the most recent restart."""
        return list(self.events)[self.idx_after_latest_restart() :]

    def init_copy(self) -> "DialogueStateTracker":
        """Creates a new state tracker with the same initial values."""
        from rasa.core.channels.channel import UserMessage

        return DialogueStateTracker(
            UserMessage.DEFAULT_SENDER_ID, self.slots.values(), self._max_event_history
        )

    def generate_all_prior_trackers(
        self,
    ) -> Generator["DialogueStateTracker", None, None]:
        """Returns a generator of the previous trackers of this tracker.

        The resulting array is representing the trackers before each action."""

        tracker = self.init_copy()

        for event in self.applied_events():

            if isinstance(event, ActionExecuted):
                yield tracker

            tracker.update(event)

        yield tracker

    def applied_events(self) -> List[Event]:
        """Returns all actions that should be applied - w/o reverted events."""

        loop_names = [
            event.name
            for event in self.events
            if isinstance(event, ActiveLoop) and event.name
        ]

        applied_events = []

        for event in self.events:
            if isinstance(event, (Restarted, SessionStarted)):
                applied_events = []
            elif isinstance(event, ActionReverted):
                self._undo_till_previous(ActionExecuted, applied_events)
            elif isinstance(event, UserUtteranceReverted):
                # Seeing a user uttered event automatically implies there was
                # a listen event right before it, so we'll first rewind the
                # user utterance, then get the action right before it (also removes
                # the `action_listen` action right before it).
                self._undo_till_previous(UserUttered, applied_events)
                self._undo_till_previous(ActionExecuted, applied_events)
            elif (
                isinstance(event, ActionExecuted)
                and event.action_name in loop_names
                and not self._first_loop_execution_or_unhappy_path(
                    event.action_name, applied_events
                )
            ):
                self._undo_till_previous_loop_execution(
                    event.action_name, applied_events
                )
            else:
                applied_events.append(event)

        return applied_events

    @staticmethod
    def _undo_till_previous(event_type: Type[Event], done_events: List[Event]) -> None:
        """Removes events from `done_events` until the first occurrence `event_type`
        is found which is also removed."""
        # list gets modified - hence we need to copy events!
        for e in reversed(done_events[:]):
            del done_events[-1]
            if isinstance(e, event_type):
                break

    def _first_loop_execution_or_unhappy_path(
        self, loop_action_name: Text, applied_events: List[Event]
    ) -> bool:
        next_action: Optional[Text] = None

        for event in reversed(applied_events):
            # Stop looking for a previous loop execution if there is a loop deactivation
            # event because it means that the current loop is running for the first
            # time and previous loop events belong to different loops.
            if isinstance(event, ActiveLoop) and event.name is None:
                return True

            if self._is_within_unhappy_path(loop_action_name, event, next_action):
                return True

            if isinstance(event, ActionExecuted):
                # We found a previous execution of the loop and we are not within an
                # unhappy path.
                if event.action_name == loop_action_name:
                    return False

                # Remember the action as we need that to check whether we might be
                # within an unhappy path.
                next_action = event.action_name

        return True

    @staticmethod
    def _is_within_unhappy_path(
        loop_action_name: Text, event: Event, next_action_in_the_future: Optional[Text]
    ) -> bool:
        # When actual users are talking to the action has to return an
        # `ActionExecutionRejected` in order to enter an unhappy path.
        loop_was_rejected_previously = (
            isinstance(event, ActionExecutionRejected)
            and event.action_name == loop_action_name
        )
        # During the policy training there are no `ActionExecutionRejected` events
        # which let us see whether we are within an unhappy path. Hence, we check if a
        # different action was executed instead of the loop after last user utterance.
        other_action_after_latest_user_utterance = (
            isinstance(event, UserUttered)
            and next_action_in_the_future is not None
            and next_action_in_the_future != loop_action_name
        )

        return loop_was_rejected_previously or other_action_after_latest_user_utterance

    @staticmethod
    def _undo_till_previous_loop_execution(
        loop_action_name: Text, done_events: List[Event]
    ) -> None:
        offset = 0
        for e in reversed(done_events[:]):
            if isinstance(e, ActionExecuted) and e.action_name == loop_action_name:
                break

            if isinstance(e, (ActionExecuted, UserUttered)):
                del done_events[-1 - offset]
            else:
                # Remember events which aren't unfeaturized to get the index right
                offset += 1

    def replay_events(self) -> None:
        """Update the tracker based on a list of events."""

        applied_events = self.applied_events()
        for event in applied_events:
            event.apply_to(self)

    def recreate_from_dialogue(self, dialogue: Dialogue) -> None:
        """Use a serialised `Dialogue` to update the trackers state.

        This uses the state as is persisted in a ``TrackerStore``. If the
        tracker is blank before calling this method, the final state will be
        identical to the tracker from which the dialogue was created."""

        if not isinstance(dialogue, Dialogue):
            raise ValueError(
                f"story {dialogue} is not of type Dialogue. "
                f"Have you deserialized it?"
            )

        self._reset()
        self.events.extend(dialogue.events)
        self.replay_events()

    def copy(self) -> "DialogueStateTracker":
        """Creates a duplicate of this tracker"""
        return self.travel_back_in_time(float("inf"))

    def travel_back_in_time(self, target_time: float) -> "DialogueStateTracker":
        """Creates a new tracker with a state at a specific timestamp.

        A new tracker will be created and all events previous to the
        passed time stamp will be replayed. Events that occur exactly
        at the target time will be included."""

        tracker = self.init_copy()

        for event in self.events:
            if event.timestamp <= target_time:
                tracker.update(event)
            else:
                break

        return tracker  # yields the final state

    def as_dialogue(self) -> Dialogue:
        """Return a ``Dialogue`` object containing all of the turns.

        This can be serialised and later used to recover the state
        of this tracker exactly."""

        return Dialogue(self.sender_id, list(self.events))

    def update(self, event: Event, domain: Optional[Domain] = None) -> None:
        """Modify the state of the tracker according to an ``Event``. """
        if not isinstance(event, Event):  # pragma: no cover
            raise ValueError("event to log must be an instance of a subclass of Event.")

        self.events.append(event)
        event.apply_to(self)

        if domain and isinstance(event, UserUttered):
            # store all entities as slots
            for e in domain.slots_for_entities(event.parse_data["entities"]):
                self.update(e)

    def as_story(self, include_source: bool = False) -> "Story":
        """Dump the tracker as a story in the Rasa Core story format.

        Returns the dumped tracker as a string."""
        from rasa.core.training.structures import Story

        story_name = (
            f"{self.sender_id} ({self.sender_source})"
            if include_source
            else self.sender_id
        )
        return Story.from_events(self.applied_events(), story_name)

    def export_stories(self, e2e: bool = False, include_source: bool = False) -> Text:
        """Dump the tracker as a story in the Rasa Core story format.

        Returns:
            The dumped tracker as a string.
        """
        # TODO: we need to revisit all usages of this, the caller needs to specify
        #       the format. this likely points to areas where we are not properly
        #       handling markdown vs yaml
        story = self.as_story(include_source)
        return story.as_story_string(flat=True, e2e=e2e)

    def export_stories_to_file(self, export_path: Text = "debug.md") -> None:
        """Dump the tracker as a story to a file."""
        import rasa.utils.io

        rasa.utils.io.write_text_file(
            self.export_stories() + "\n", export_path, append=True
        )

    def get_last_event_for(
        self,
        event_type: Type[Event],
        action_names_to_exclude: List[Text] = None,
        skip: int = 0,
        event_verbosity: EventVerbosity = EventVerbosity.APPLIED,
    ) -> Optional[Event]:
        """Gets the last event of a given type which was actually applied.

        Args:
            event_type: The type of event you want to find.
            action_names_to_exclude: Events of type `ActionExecuted` which
                should be excluded from the results. Can be used to skip
                `action_listen` events.
            skip: Skips n possible results before return an event.
            event_verbosity: Which `EventVerbosity` should be used to search for events.

        Returns:
            event which matched the query or `None` if no event matched.
        """

        to_exclude = action_names_to_exclude or []

        def filter_function(e: Event):
            has_instance = isinstance(e, event_type)
            excluded = isinstance(e, ActionExecuted) and e.action_name in to_exclude
            return has_instance and not excluded

        filtered = filter(
            filter_function, reversed(self._events_for_verbosity(event_verbosity) or [])
        )

        for i in range(skip):
            next(filtered, None)

        return next(filtered, None)

    def last_executed_action_has(self, name: Text, skip: int = 0) -> bool:
        """Returns whether last `ActionExecuted` event had a specific name.

        Args:
            name: Name of the event which should be matched.
            skip: Skips n possible results in between.

        Returns:
            `True` if last executed action had name `name`, otherwise `False`.
        """

        last: Optional[ActionExecuted] = self.get_last_event_for(
            ActionExecuted, action_names_to_exclude=[ACTION_LISTEN_NAME], skip=skip
        )
        return last is not None and last.action_name == name

    ###
    # Internal methods for the modification of the trackers state. Should
    # only be called by events, not directly. Rather update the tracker
    # with an event that in its ``apply_to`` method modifies the tracker.
    ###
    def _reset(self) -> None:
        """Reset tracker to initial state - doesn't delete events though!."""

        self._reset_slots()
        self._paused = False
        self.latest_action_name = None
        self.latest_message = UserUttered.empty()
        self.latest_bot_utterance = BotUttered.empty()
        self.followup_action = ACTION_LISTEN_NAME
        self.active_loop = {}

    def _reset_slots(self) -> None:
        """Set all the slots to their initial value."""

        for slot in self.slots.values():
            slot.reset()

    def _set_slot(self, key: Text, value: Any) -> None:
        """Set the value of a slot if that slot exists."""

        if key in self.slots:
            self.slots[key].value = value
        else:
            logger.error(
                f"Tried to set non existent slot '{key}'. Make sure you "
                f"added all your slots to your domain file."
            )

    def _create_events(self, evts: List[Event]) -> Deque[Event]:

        if evts and not isinstance(evts[0], Event):  # pragma: no cover
            raise ValueError("events, if given, must be a list of events")
        return deque(evts, self._max_event_history)

    def __eq__(self, other) -> bool:
        if isinstance(self, type(other)):
            return other.events == self.events and self.sender_id == other.sender_id
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def trigger_followup_action(self, action: Text) -> None:
        """Triggers another action following the execution of the current."""

        self.followup_action = action

    def clear_followup_action(self) -> None:
        """Clears follow up action when it was executed."""

        self.followup_action = None

    def _merge_slots(
        self, entities: Optional[List[Dict[Text, Any]]] = None
    ) -> List[SlotSet]:
        """Take a list of entities and create tracker slot set events.

        If an entity type matches a slots name, the entities value is set
        as the slots value by creating a ``SlotSet`` event.
        """

        entities = entities if entities else self.latest_message.entities
        new_slots = [
            SlotSet(e["entity"], e["value"])
            for e in entities
            if e["entity"] in self.slots.keys()
        ]
        return new_slots

    def active_loop_name(self) -> Optional[Text]:
        """Get the name of the currently active loop.

        Returns: `None` if no active loop or the name of the currently active loop.
        """
        if not self.active_loop:
            return None

        return self.active_loop.get("name")
