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
from rasa_core.conversation import Dialogue
from rasa_core.events import UserUttered, TopicSet, ActionExecuted, \
    Event, SlotSet, Restarted, ActionReverted, UserUtteranceReverted

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.actions import Action


class DialogueStateTracker(object):
    """Maintains the state of a conversation."""

    def __init__(self, sender_id, slots,
                 topics=None,
                 default_topic=None,
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
        # available topics in the domain
        self.topics = topics if topics is not None else []
        # default topic of the domain
        self.default_topic = default_topic
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
        # topic tracking
        self._topic_stack = None
        self.latest_action_name = None
        self.latest_message = None
        self.latest_restart_event = None
        self._reset()

    ###
    # Public tracker interface
    ###
    def current_state(self):
        # type: () -> Dict[Text, Any]
        """Returns the current tracker state as an object."""

        return {
            "sender_id": self.sender_id,
            "slots": self.current_slot_values(),
            "latest_message": self.latest_message.parse_data
        }

    def current_slot_values(self):
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
        """States whether the tracker is currently paused."""
        return self._paused

    def _idx_after_latest_restart(self):
        if self.latest_restart_event is not None:
            return self.latest_restart_event
        else:
            return 0

    def _events_after_latest_restart(self):
        return list(self.events)[self._idx_after_latest_restart():]

    @property
    def previous_topic(self):
        # type: () -> Optional[Text]
        """Retrieves the topic that was set before the current one."""

        for event in reversed(self._events_after_latest_restart()):
            if isinstance(event, TopicSet):
                return event.topic
        return None

    @property
    def topic(self):
        # type: () -> Text
        """Retrieves current topic, or default if no topic has been set yet."""

        return self._topic_stack.top

    def generate_all_prior_states(self):
        # type: () -> Generator[DialogueStateTracker, None, None]
        """Returns a generator of the previous states of this tracker.

        The resulting array is representing the state before each action."""
        from rasa_core.channels import UserMessage

        tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                       self.slots.values(),
                                       self.topics,
                                       self.default_topic)

        for event in self._applied_events():
            if isinstance(event, ActionExecuted):
                yield tracker
            tracker.update(event)

        yield tracker  # yields the final state

    def _applied_events(self):
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
                undo_till_previous(UserUttered, applied_events)
            else:
                applied_events.append(event)
        return applied_events

    def replay_events(self):
        # type: (int) -> None
        """Update the tracker based on a list of events."""
        applied_events = self._applied_events()
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
        from rasa_core.training_utils.dsl import StoryStep, Story

        story_step = StoryStep()
        for event in self._applied_events():
            story_step.add_event(event)
        story = Story([story_step])
        return story.as_story_string(flat=True)

    def export_stories_to_file(self, export_path="debug.md"):
        with io.open(export_path, 'a') as f:
            f.write(self.export_stories())

    ###
    # Internal methods for the modification of the trackers state. Should
    # only be called by events, not directly. Rather update the tracker
    # with an event that in its ``apply_on`` method modifies the tracker.
    ###
    def _reset(self):
        # type: () -> None
        """Reset tracker to initial state - doesn't delete events though!."""

        self._reset_slots()
        self._paused = False
        self.latest_action_name = None
        self.latest_message = UserUttered.empty()
        self.follow_up_action = None
        self._topic_stack = utils.TopicStack(self.topics, [],
                                             self.default_topic)

    def _reset_slots(self):
        for slot in self.slots.values():
            slot.reset()

    def _set_slot(self, key, value):
        if key in self.slots:
            self.slots[key].value = value
        else:
            logger.warn("Tried to set non existent slot '{}'".format(key))

    def _create_events(self, events):
        # type: (List[Event]) -> deque

        if events and not isinstance(events[0], Event):  # pragma: no cover
            raise ValueError("events, if given, must be a list of events")
        return deque(events, self._max_event_history)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            other_encoded = jsonpickle.encode(other.as_dialogue())
            encoded = jsonpickle.encode(self.as_dialogue())
            return (other_encoded == encoded and
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
        entities = entities if entities else self.latest_message.entities
        new_slots = [SlotSet(e["entity"], e["value"]) for e in entities if
                     e["entity"] in self.slots.keys()]
        return new_slots
