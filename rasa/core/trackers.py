import copy
import logging
from collections import deque, defaultdict
from enum import Enum
from typing import Dict, Text, Any, Optional, Iterator, Type, List

from rasa.core import events
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.conversation import Dialogue
from rasa.core.events import (
    UserUttered,
    ActionExecuted,
    Event,
    SlotSet,
    Restarted,
    ActionReverted,
    UserUtteranceReverted,
    BotUttered,
    Form,
)
from rasa.core.domain import Domain
from rasa.core.slots import Slot

logger = logging.getLogger(__name__)


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

    def __missing__(self, key):
        value = self[key] = Slot(key)
        return value


class DialogueStateTracker(object):
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
    ):
        tracker = cls(sender_id, slots, max_event_history)
        for e in evts:
            tracker.update(e)
        return tracker

    def __init__(self, sender_id, slots, max_event_history=None):
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
            self.slots = {slot.name: copy.deepcopy(slot) for slot in slots}
        else:
            self.slots = AnySlotDict()

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
        self.active_form = {}

    ###
    # Public tracker interface
    ###
    def current_state(
        self, event_verbosity: EventVerbosity = EventVerbosity.NONE
    ) -> Dict[Text, Any]:
        """Return the current tracker state as an object."""

        if event_verbosity == EventVerbosity.ALL:
            evts = [e.as_dict() for e in self.events]
        elif event_verbosity == EventVerbosity.AFTER_RESTART:
            evts = [e.as_dict() for e in self.events_after_latest_restart()]
        elif event_verbosity == EventVerbosity.APPLIED:
            evts = [e.as_dict() for e in self.applied_events()]
        else:
            evts = None

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
            "events": evts,
            "latest_input_channel": self.get_latest_input_channel(),
            "active_form": self.active_form,
            "latest_action_name": self.latest_action_name,
        }

    def past_states(self, domain) -> deque:
        """Generate the past states of this tracker based on the history."""

        generated_states = domain.states_for_tracker_history(self)
        return deque((frozenset(s.items()) for s in generated_states))

    def change_form_to(self, form_name: Text) -> None:
        """Activate or deactivate a form"""
        if form_name is not None:
            self.active_form = {
                "name": form_name,
                "validate": True,
                "rejected": False,
                "trigger_message": self.latest_message.parse_data,
            }
        else:
            self.active_form = {}

    def set_form_validation(self, validate: bool) -> None:
        """Toggle form validation"""
        self.active_form["validate"] = validate

    def reject_action(self, action_name: Text) -> None:
        """Notify active form that it was rejected"""
        if action_name == self.active_form.get("name"):
            self.active_form["rejected"] = True

    def set_latest_action_name(self, action_name: Text) -> None:
        """Set latest action name
            and reset form validation and rejection parameters
        """
        self.latest_action_name = action_name
        if self.active_form.get("name"):
            # reset form validation if some form is active
            self.active_form["validate"] = True
        if action_name == self.active_form.get("name"):
            # reset form rejection if it was predicted again
            self.active_form["rejected"] = False

    def current_slot_values(self) -> [Dict[Text, Any]]:
        """Return the currently set values of the slots"""
        return {key: slot.value for key, slot in self.slots.items()}

    def get_slot(self, key: Text) -> Optional[Any]:
        """Retrieves the value of a slot."""

        if key in self.slots:
            return self.slots[key].value
        else:
            logger.info("Tried to access non existent slot '{}'".format(key))
            return None

    def get_latest_entity_values(self, entity_type: Text) -> Iterator[Text]:
        """Get entity values found for the passed entity name in latest msg.

        If you are only interested in the first entity of a given type use
        `next(tracker.get_latest_entity_values("my_entity_name"), None)`.
        If no entity is found `None` is the default result."""

        return (
            x.get("value")
            for x in self.latest_message.entities
            if x.get("entity") == entity_type
        )

    def get_latest_input_channel(self) -> Optional[Text]:
        """Get the name of the input_channel of the latest UserUttered event"""

        for e in reversed(self.events):
            if isinstance(e, UserUttered):
                return e.input_channel

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

    def init_copy(self):
        # type: () -> DialogueStateTracker
        """Creates a new state tracker with the same initial values."""
        from rasa.core.channels import UserMessage

        return DialogueStateTracker(
            UserMessage.DEFAULT_SENDER_ID, self.slots.values(), self._max_event_history
        )

    def generate_all_prior_trackers(self):
        # type: () -> Generator[DialogueStateTracker, None, None]
        """Returns a generator of the previous trackers of this tracker.

        The resulting array is representing
        the trackers before each action."""

        tracker = self.init_copy()

        ignored_trackers = []
        latest_message = tracker.latest_message

        for i, event in enumerate(self.applied_events()):
            if isinstance(event, UserUttered):
                if tracker.active_form.get("name") is None:
                    # store latest user message before the form
                    latest_message = event

            elif isinstance(event, Form):
                # form got either activated or deactivated, so override
                # tracker's latest message
                tracker.latest_message = latest_message

            elif isinstance(event, ActionExecuted):
                # yields the intermediate state
                if tracker.active_form.get("name") is None:
                    yield tracker

                elif tracker.active_form.get("rejected"):
                    for tr in ignored_trackers:
                        yield tr
                    ignored_trackers = []

                    if not tracker.active_form.get(
                        "validate"
                    ) or event.action_name != tracker.active_form.get("name"):
                        # persist latest user message
                        # that was rejected by the form
                        latest_message = tracker.latest_message
                    else:
                        # form was called with validation, so
                        # override tracker's latest message
                        tracker.latest_message = latest_message

                    yield tracker

                elif event.action_name != tracker.active_form.get("name"):
                    # it is not known whether the form will be
                    # successfully executed, so store this tracker for later
                    tr = tracker.copy()
                    # form was called with validation, so
                    # override tracker's latest message
                    tr.latest_message = latest_message
                    ignored_trackers.append(tr)

                if event.action_name == tracker.active_form.get("name"):
                    # the form was successfully executed, so
                    # remove all stored trackers
                    ignored_trackers = []

            tracker.update(event)

        # yields the final state
        if tracker.active_form.get("name") is None:
            yield tracker
        elif tracker.active_form.get("rejected"):
            for tr in ignored_trackers:
                yield tr
            yield tracker

    def applied_events(self) -> List[Event]:
        """Returns all actions that should be applied - w/o reverted events."""

        def undo_till_previous(event_type, done_events):
            """Removes events from `done_events` until `event_type` is
               found."""
            # list gets modified - hence we need to copy events!
            for e in reversed(done_events[:]):
                del done_events[-1]
                if isinstance(e, event_type):
                    break

        applied_events = []
        for event in self.events:
            if isinstance(event, Restarted):
                applied_events = []
            elif isinstance(event, ActionReverted):
                undo_till_previous(ActionExecuted, applied_events)
            elif isinstance(event, UserUtteranceReverted):
                # Seeing a user uttered event automatically implies there was
                # a listen event right before it, so we'll first rewind the
                # user utterance, then get the action right before it (the
                # listen action).
                undo_till_previous(UserUttered, applied_events)
                undo_till_previous(ActionExecuted, applied_events)
            else:
                applied_events.append(event)
        return applied_events

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
                "story {0} is not of type Dialogue. "
                "Have you deserialized it?".format(dialogue)
            )

        self._reset()
        self.events.extend(dialogue.events)
        self.replay_events()

    def copy(self):
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

    def export_stories(self, e2e=False) -> Text:
        """Dump the tracker as a story in the Rasa Core story format.

        Returns the dumped tracker as a string."""
        from rasa.core.training.structures import Story

        story = Story.from_events(self.applied_events(), self.sender_id)
        return story.as_story_string(flat=True, e2e=e2e)

    def export_stories_to_file(self, export_path: Text = "debug.md") -> None:
        """Dump the tracker as a story to a file."""

        with open(export_path, "a", encoding="utf-8") as f:
            f.write(self.export_stories() + "\n")

    def get_last_event_for(
        self,
        event_type: Type[Event],
        action_names_to_exclude: List[Text] = None,
        skip: int = 0,
    ) -> Optional[Event]:
        """Gets the last event of a given type which was actually applied.

        Args:
            event_type: The type of event you want to find.
            action_names_to_exclude: Events of type `ActionExecuted` which
                should be excluded from the results. Can be used to skip
                `action_listen` events.
            skip: Skips n possible results before return an event.

        Returns:
            event which matched the query or `None` if no event matched.
        """

        to_exclude = action_names_to_exclude or []

        def filter_function(e: Event):
            has_instance = isinstance(e, event_type)
            excluded = isinstance(e, ActionExecuted) and e.action_name in to_exclude

            return has_instance and not excluded

        filtered = filter(filter_function, reversed(self.applied_events()))
        for i in range(skip):
            next(filtered, None)

        return next(filtered, None)

    def last_executed_action_has(self, name: Text, skip=0) -> bool:
        """Returns whether last `ActionExecuted` event had a specific name.

        Args:
            name: Name of the event which should be matched.
            skip: Skips n possible results in between.

        Returns:
            `True` if last executed action had name `name`, otherwise `False`.
        """

        last = self.get_last_event_for(
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
        self.active_form = {}

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
                "Tried to set non existent slot '{}'. Make sure you "
                "added all your slots to your domain file."
                "".format(key)
            )

    def _create_events(self, evts: List[Event]) -> deque:

        if evts and not isinstance(evts[0], Event):  # pragma: no cover
            raise ValueError("events, if given, must be a list of events")
        return deque(evts, self._max_event_history)

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return other.events == self.events and self.sender_id == other.sender_id
        else:
            return False

    def __ne__(self, other):
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
