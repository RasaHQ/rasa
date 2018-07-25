from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import io
import logging
from collections import deque

import jsonpickle
import typing
from typing import Generator, Dict, Text, Any, Optional, Iterator
from typing import List

from rasa_core import utils
from rasa_core import events
from rasa_core.conversation import Dialogue
from rasa_core.events import UserUttered, ActionExecuted, \
    Event, SlotSet, Restarted, ActionReverted, UserUtteranceReverted, \
    BotUttered, TopicSet

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.actions import Action
    from rasa_core.domain import Domain


class DialogueStateTracker(object):
    """Maintains the state of a conversation."""

    @classmethod
    def from_dict(cls, sender_id, dump_as_dict, domain):
        # type: (Text, List[Dict[Text, Any]]) -> DialogueStateTracker
        """Create a tracker from dump.

        The dump should be an array of dumped events. When restoring
        the tracker, these events will be replayed to recreate the state."""

        evts = events.deserialise_events(dump_as_dict)
        tracker = cls(sender_id, domain.slots)
        for e in evts:
            tracker.update(e)
        return tracker

    def __init__(self, sender_id, slots,
                 max_event_history=None):
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
        self.slots = {slot.name: copy.deepcopy(slot) for slot in slots}

        ###
        # current state of the tracker - MUST be re-creatable by processing
        # all the events. This only defines the attributes, values are set in
        # `reset()`
        ###
        # if tracker is paused, no actions should be taken
        self._paused = None
        # A deterministically scheduled action to be executed next
        self.follow_up_action = None
        self.latest_action_name = None
        self.latest_message = None
        # Stores the most recent message sent by the user
        self.latest_bot_utterance = None
        self._reset()

    ###
    # Public tracker interface
    ###
    def current_state(self,
                      should_include_events=False,
                      only_events_after_latest_restart=False):
        # type: (bool, bool) -> Dict[Text, Any]
        """Return the current tracker state as an object."""

        if should_include_events:
            if only_events_after_latest_restart:
                es = self.events
            else:
                es = self.events_after_latest_restart()
            evts = [e.as_dict() for e in es]
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
            "paused": self.is_paused(),
            "events": evts
        }

    def past_states(self, domain):
        # type: (Domain) -> deque
        """Generate the past states of this tracker based on the history."""

        generated_states = domain.states_for_tracker_history(self)
        return deque((frozenset(s.items()) for s in generated_states))

    def current_slot_values(self):
        # type: () -> Dict[Text, Any]
        """Return the currently set values of the slots"""
        return {key: slot.value for key, slot in self.slots.items()}

    def get_slot(self, key):
        # type: (Text) -> Optional[Any]
        """Retrieves the value of a slot."""

        if key in self.slots:
            return self.slots[key].value
        else:
            logger.info("Tried to access non existent slot '{}'".format(key))
            return None

    def get_latest_entity_values(self, entity_type):
        # type: (Text) -> Iterator[Text]
        """Get entity values found for the passed entity name in latest msg.

        If you are only interested in the first entity of a given type use
        `next(tracker.get_latest_entity_values("my_entity_name"), None)`.
        If no entity is found `None` is the default result."""

        return (x.get("value")
                for x in self.latest_message.entities
                if x.get("entity") == entity_type)

    def is_paused(self):
        # type: () -> bool
        """State whether the tracker is currently paused."""
        return self._paused

    def idx_after_latest_restart(self):
        # type: () -> int
        """Return the idx of the most recent restart in the list of events.

        If the conversation has not been restarted, ``0`` is returned."""

        idx = 0
        for i, event in enumerate(self.events):
            if isinstance(event, Restarted):
                idx = i + 1
        return idx

    def events_after_latest_restart(self):
        # type: () -> List[Event]
        """Return a list of events after the most recent restart."""
        return list(self.events)[self.idx_after_latest_restart():]

    def init_copy(self):
        # type: () -> DialogueStateTracker
        """Creates a new state tracker with the same initial values."""
        from rasa_core.channels import UserMessage

        return DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                    self.slots.values(),
                                    self._max_event_history)

    def generate_all_prior_trackers(self):
        # type: () -> Generator[DialogueStateTracker, None, None]
        """Returns a generator of the previous trackers of this tracker.

        The resulting array is representing
        the trackers before each action."""

        tracker = self.init_copy()

        for event in self.applied_events():
            if isinstance(event, ActionExecuted):
                yield tracker
            tracker.update(event)

        yield tracker  # yields the final state

    def applied_events(self):
        # type: () -> List[Event]
        """Returns all actions that should be applied - w/o reverted events."""
        def undo_till_previous(event_type, done_events):
            """Removes events from `done_events` until `event_type` is found."""
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
            elif isinstance(event, TopicSet):
                logger.warn("Topics are deprecated, therefore the TopicSet "
                            "event will be ignored")
            else:
                applied_events.append(event)
        return applied_events

    def replay_events(self):
        # type: () -> None
        """Update the tracker based on a list of events."""

        applied_events = self.applied_events()
        for event in applied_events:
            event.apply_to(self)

    def recreate_from_dialogue(self, dialogue):
        # type: (Dialogue) -> None
        """Use a serialised `Dialogue` to update the trackers state.

        This uses the state as is persisted in a ``TrackerStore``. If the
        tracker is blank before calling this method, the final state will be
        identical to the tracker from which the dialogue was created."""

        if not isinstance(dialogue, Dialogue):
            raise ValueError("story {0} is not of type Dialogue. "
                             "Have you deserialized it?".format(dialogue))

        self._reset()
        self.events.extend(dialogue.events)
        self.replay_events()

    def copy(self):
        """Creates a duplicate of this tracker"""
        return self.travel_back_in_time(float("inf"))

    def travel_back_in_time(self, target_time):
        # type: (float) -> DialogueStateTracker
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

    def as_dialogue(self):
        # type: () -> Dialogue
        """Return a ``Dialogue`` object containing all of the turns.

        This can be serialised and later used to recover the state
        of this tracker exactly."""

        return Dialogue(self.sender_id, list(self.events))

    def update(self, event):
        # type: (Event) -> None
        """Modify the state of the tracker according to an ``Event``. """

        if not isinstance(event, Event):  # pragma: no cover
            raise ValueError("event to log must be an instance "
                             "of a subclass of Event.")

        self.events.append(event)
        event.apply_to(self)

    def export_stories(self):
        # type: () -> Text
        """Dump the tracker as a story in the Rasa Core story format.

        Returns the dumped tracker as a string."""
        from rasa_core.training.structures import Story

        story = Story.from_events(self.applied_events())
        return story.as_story_string(flat=True)

    def export_stories_to_file(self, export_path="debug.md"):
        # type: (Text) -> None
        """Dump the tracker as a story to a file."""

        with io.open(export_path, 'a') as f:
            f.write(self.export_stories() + "\n")

    ###
    # Internal methods for the modification of the trackers state. Should
    # only be called by events, not directly. Rather update the tracker
    # with an event that in its ``apply_to`` method modifies the tracker.
    ###
    def _reset(self):
        # type: () -> None
        """Reset tracker to initial state - doesn't delete events though!."""

        self._reset_slots()
        self._paused = False
        self.latest_action_name = None
        self.latest_message = UserUttered.empty()
        self.latest_bot_utterance = BotUttered.empty()
        self.follow_up_action = None

    def _reset_slots(self):
        # type: () -> None
        """Set all the slots to their initial value."""

        for slot in self.slots.values():
            slot.reset()

    def _set_slot(self, key, value):
        # type: (Text, Any) -> None
        """Set the value of a slot if that slot exists."""

        if key in self.slots:
            self.slots[key].value = value
        else:
            logger.error("Tried to set non existent slot '{}'. Make sure you "
                         "added all your slots to your domain file."
                         "".format(key))

    def _create_events(self, evts):
        # type: (List[Event]) -> deque

        if evts and not isinstance(evts[0], Event):  # pragma: no cover
            raise ValueError("events, if given, must be a list of events")
        return deque(evts, self._max_event_history)

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return (other.events == self.events and
                    self.sender_id == other.sender_id)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def trigger_follow_up_action(self, action):
        # type: (Action) -> None
        """Triggers another action following the execution of the current."""

        self.follow_up_action = action

    def clear_follow_up_action(self):
        # type: () -> None
        """Clears follow up action when it was executed"""

        self.follow_up_action = None

    def _merge_slots(self, entities=None):
        # type: (Optional[List[Dict[Text, Any]]]) -> List[SlotSet]
        """Take a list of entities and create tracker slot set events.

        If an entity type matches a slots name, the entities value is set
        as the slots value by creating a ``SlotSet`` event."""

        entities = entities if entities else self.latest_message.entities
        new_slots = [SlotSet(e["entity"], e["value"]) for e in entities if
                     e["entity"] in self.slots.keys()]
        return new_slots
