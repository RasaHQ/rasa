from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import uuid

import typing
from builtins import str

from rasa_core import utils
from rasa_core.actions.action import ACTION_LISTEN_NAME

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember
class Event(object):
    """An event is one of the following:
    - something the user has said to the bot (starts a new turn)
    - the topic has been set
    - the bot has taken an action

    Events are logged by the Tracker's log_event method.
    This updates the list of turns so that the current state
    can be recovered by consuming the list of turns."""

    type_name = "event"

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    def as_story_string(self):
        raise NotImplementedError

    @staticmethod
    def from_story_string(event_name, parameters, domain, default=None):
        event = Event.resolve_by_type(event_name, default)
        return event._from_story_string(event_name, parameters, domain)

    @staticmethod
    def from_parameters(event_name, parameters, domain, default=None):
        event = Event.resolve_by_type(event_name, default)
        return event._from_parameters(event_name, parameters, domain)

    @classmethod
    def _from_story_string(cls, event_name, parameters, domain):
        """Called to convert a parsed story line into an event."""
        return cls()

    @classmethod
    def _from_parameters(cls, event_name, parameters, domain):
        """Called to convert a dictionary of parameters to an event.

        By default uses the same implementation as the story line
        conversation ``_from_story_string``. But the subclass might
        decide to handle parameters differently if the parsed parameters
        don't origin from a story file."""

        return cls._from_story_string(event_name, parameters, domain)

    @staticmethod
    def resolve_by_type(type_name, default=None):
        """Returns a slots class by its type name."""

        for cls in utils.all_subclasses(Event):
            if cls.type_name == type_name:
                return cls
        if default is not None:
            return default
        else:
            raise ValueError("Unknown event name '{}'.".format(type_name))

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None
        pass


# noinspection PyProtectedMember
class UserUttered(Event):
    """The user has said something to the bot.

    As a side effect a new ``Turn`` will be created in the ``Tracker``."""

    type_name = "user"

    def __init__(self, text, intent=None, entities=None, parse_data=None):
        self.text = text
        self.intent = intent if intent else {}
        self.entities = entities if entities else []

        if parse_data:
            self.parse_data = parse_data
        else:
            self.parse_data = {
                "intent": self.intent,
                "entities": self.entities,
                "text": text}

    @staticmethod
    def from_parse_data(text, parse_data):
        return UserUttered(text, parse_data["intent"], parse_data["entities"],
                           parse_data)

    def __hash__(self):
        return hash((self.text, self.intent, tuple(self.entities)))

    def __eq__(self, other):
        if not isinstance(other, UserUttered):
            return False
        else:
            return (self.text, self.intent, self.entities, self.parse_data) == \
                   (other.text, other.intent, other.entities, other.parse_data)

    def __str__(self):
        return ("UserUttered(text: {}, intent: {}, "
                "entities: {})".format(self.text, self.intent, self.entities))

    @staticmethod
    def empty():
        return UserUttered(None)

    def as_story_string(self):
        if self.intent:
            if self.entities:
                entity_strs = ['{}={}'.format(ent['entity'], ent['value'])
                               for ent in self.entities]
                ent_string = "[" + ",".join(entity_strs) + "]"
            else:
                ent_string = ""

            return "_{intent}{entities}".format(
                    intent=self.intent.get("name", ""),
                    entities=ent_string)
        else:
            return self.text

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None

        tracker.latest_message = self


# noinspection PyProtectedMember
class TopicSet(Event):
    """The topic of conversation has changed.

    As a side effect self.topic will be pushed on to ``Tracker.topic_stack``."""

    type_name = "topic"

    def __init__(self, topic):
        self.topic = topic

    def __str__(self):
        return "TopicSet(topic: {})".format(self.topic)

    def __hash__(self):
        return hash(self.topic)

    def __eq__(self, other):
        if not isinstance(other, TopicSet):
            return False
        else:
            return self.topic == other.topic

    def as_story_string(self):
        return "{name}[{props}]".format(name=self.type_name, props=self.topic)

    @classmethod
    def _from_story_string(cls, event_name, parameters, domain):
        topic = list(parameters.keys())[0] if parameters else ""
        return TopicSet(topic)

    @classmethod
    def _from_parameters(cls, event_name, parameters, domain):
        try:
            return TopicSet(parameters["topic"])
        except KeyError as e:
            raise ValueError("Failed to parse set topic event. {}".format(e))

    def apply_to(self, tracker):
        tracker._topic_stack.push(self.topic)


# noinspection PyProtectedMember
class SlotSet(Event):
    """The user has specified their preference for the value of a ``slot``.

    As a side effect the ``Tracker``'s slots will be updated so
    that ``tracker.slots[key]=value``."""

    type_name = "slot"

    def __init__(self, key, value=None):
        self.key = key
        self.value = value

    def __str__(self):
        return "SlotSet(key: {}, value: {})".format(self.key, self.value)

    def __hash__(self):
        return hash((self.key, self.value))

    def __eq__(self, other):
        if not isinstance(other, SlotSet):
            return False
        else:
            return (self.key, self.value) == (other.key, other.value)

    def as_story_string(self):
        props = json.dumps({self.key: self.value})
        return "{name}{props}".format(name=self.type_name, props=props)

    @classmethod
    def _from_story_string(cls, event_name, parameters, domain):
        slot_key = list(parameters.keys())[0] if parameters else None
        if slot_key:
            return SlotSet(slot_key, parameters[slot_key])
        else:
            return None

    @classmethod
    def _from_parameters(cls, event_name, parameters, domain):
        try:
            return SlotSet(parameters["name"], parameters["value"])
        except KeyError as e:
            raise ValueError("Failed to parse set slot event. {}".format(e))

    def apply_to(self, tracker):
        tracker._set_slot(self.key, self.value)


# noinspection PyProtectedMember
class Restarted(Event):
    """Conversation should start over & history wiped.

    As a side effect the ``Tracker`` will be reinitialised."""

    type_name = "restart"

    def __hash__(self):
        return hash(32143124312)

    def __eq__(self, other):
        return isinstance(other, Restarted)

    def __str__(self):
        return "Restarted()"

    def as_story_string(self):
        return self.type_name

    def apply_to(self, tracker):
        tracker._reset()
        tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
        tracker.latest_restart_event = len(tracker.events)


# noinspection PyProtectedMember
class UserUtteranceReverted(Event):
    """Bot undoes its last action.

    Shouldn't be used during actual user interactions, mostly for train.
    As a side effect the ``Tracker``'s last turn is removed."""

    type_name = "rewind"

    def __hash__(self):
        return hash(32143124315)

    def __eq__(self, other):
        return isinstance(other, UserUtteranceReverted)

    def __str__(self):
        return "UserUtteranceReverted()"

    def as_story_string(self):
        return self.type_name

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None

        tracker._reset()
        tracker.replay_events()


# noinspection PyProtectedMember
class AllSlotsReset(Event):
    """Conversation should start over & history wiped.

    As a side effect the ``Tracker`` will be reinitialised."""

    type_name = "reset_slots"

    def __hash__(self):
        return hash(32143124316)

    def __eq__(self, other):
        return isinstance(other, AllSlotsReset)

    def __str__(self):
        return "AllSlotsReset()"

    def as_story_string(self):
        return self.type_name

    def apply_to(self, tracker):
        tracker._reset_slots()


# noinspection PyProtectedMember
class ReminderScheduled(Event):
    """ Allows asynchronous scheduling of action execution.

    As a side effect the message processor will schedule an action to be run
    at the trigger date."""

    type_name = "reminder"

    def __init__(self, action_name, trigger_date_time, name=None,
                 kill_on_user_message=True):
        """Creates the reminder

        :param action_name: name of the action to be scheduled
        :param trigger_date_time: date at which the execution of the action
                                  should be triggered
        :param name: id of the reminder. if there are multiple reminders with
                     the same id only the last will be run
        :param kill_on_user_message: ``True`` means a user message before the
                                     trigger date will abort the reminder
        """

        self.action_name = action_name
        self.trigger_date_time = trigger_date_time
        self.kill_on_user_message = kill_on_user_message
        self.name = name if name is not None else str(uuid.uuid1())

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, ReminderScheduled):
            return False
        else:
            return self.name == other.name

    def __str__(self):
        return ("ReminderScheduled("
                "action: {}, trigger_date: {}, name: {}"
                ")".format(self.action_name, self.trigger_date_time, self.name))

    def as_story_string(self):
        props = json.dumps({
            "action": self.action_name,
            "date_time": self.trigger_date_time,
            "name": self.name,
            "kill_on_user_msg": self.kill_on_user_message})
        return "{name}{props}".format(name=self.type_name, props=props)

    @classmethod
    def _from_story_string(cls, event_name, parameters, domain):
        logger.info("Reminders will be ignored during training, "
                    "which should be ok.")
        return ReminderScheduled(parameters["action"],
                                 parameters["date_time"],
                                 parameters.get("name", None),
                                 parameters.get("kill_on_user_msg", True))


# noinspection PyProtectedMember
class ActionReverted(Event):
    """Bot undoes its last action.

    Shouldn't be used during actual user interactions, mostly for train.
    As a side effect the ``Tracker``'s last turn is removed."""

    type_name = "undo"

    def __hash__(self):
        return hash(32143124318)

    def __eq__(self, other):
        return isinstance(other, ActionReverted)

    def __str__(self):
        return "ActionReverted()"

    def as_story_string(self):
        return self.type_name

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None

        tracker._reset()
        tracker.replay_events()


# noinspection PyProtectedMember
class StoryExported(Event):
    """Story should get dumped to a file."""

    type_name = "export"

    def __init__(self, path=None):
        self.path = path if path else "stories.md"

    def __hash__(self):
        return hash(32143124319)

    def __eq__(self, other):
        return isinstance(other, StoryExported)

    def __str__(self):
        return "StoryExported()"

    def as_story_string(self):
        return self.type_name

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None

        tracker.export_stories_to_file(self.path)


# noinspection PyProtectedMember
class ConversationPaused(Event):
    """Ignore messages from the user to let a human take over.

    As a side effect the ``Tracker``'s ``paused`` attribute will
    be set to ``True``. """

    type_name = "pause"

    def __hash__(self):
        return hash(32143124313)

    def __eq__(self, other):
        return isinstance(other, ConversationPaused)

    def __str__(self):
        return "ConversationPaused()"

    def as_story_string(self):
        return self.type_name

    def apply_to(self, tracker):
        tracker._paused = True


# noinspection PyProtectedMember
class ConversationResumed(Event):
    """Bot takes over conversation.

    Inverse of ``PauseConversation``. As a side effect the ``Tracker``'s
    ``paused`` attribute will be set to ``False``."""

    type_name = "resume"

    def __hash__(self):
        return hash(32143124314)

    def __eq__(self, other):
        return isinstance(other, ConversationResumed)

    def __str__(self):
        return "ConversationResumed()"

    def as_story_string(self):
        return self.type_name

    def apply_to(self, tracker):
        tracker._paused = False


# noinspection PyProtectedMember
class ActionExecuted(Event):
    """An operation describes an action taken + its result.

    It comprises an action and a list of events. operations will be appended
    to the latest ``Turn`` in the ``Tracker.turns``."""

    def __init__(self, action_name):
        self.action_name = action_name
        self.unpredictable = False

    def __str__(self):
        return "ActionExecuted(action: {})".format(self.action_name)

    def __hash__(self):
        return hash(self.action_name)

    def __eq__(self, other):
        if not isinstance(other, ActionExecuted):
            return False
        else:
            return self.action_name == other.action_name

    def as_story_string(self):
        return self.action_name

    @classmethod
    def _from_story_string(cls, event_name, parameters, domain):
        if event_name in domain.action_names:
            return ActionExecuted(event_name)
        else:
            return None

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None

        tracker.latest_action_name = self.action_name
