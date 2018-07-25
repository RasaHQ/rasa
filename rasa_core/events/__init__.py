from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import json
import logging
import time
import uuid

import jsonpickle
import typing
from builtins import str
from typing import List, Dict, Text, Any

from rasa_core import utils

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


def deserialise_events(serialized_events):
    # type: (List[Dict[Text, Any]]) -> List[Event]
    """Convert a list of dictionaries to a list of corresponding events.

    Example format:
        [{"event": "set_slot", "value": 5, "name": "my_slot"}]
    """

    return [Event.from_parameters(e)
            for e in serialized_events
            if "event" in e]


def first_key(d, default_key):
    if len(d) > 1:
        for k, v in d.items():
            if k != default_key:
                # we return the first key that is not the default key
                return k
    elif len(d) == 1:
        return list(d.keys())[0]
    else:
        return None


# noinspection PyProtectedMember
class Event(object):
    """Events describe everything that occurs in 
    a conversation and tell the :class:`DialogueStateTracker`
    how to update its state."""

    type_name = "event"

    def __init__(self, timestamp=None):
        self.timestamp = timestamp if timestamp else time.time()

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    def as_story_string(self):
        raise NotImplementedError

    @staticmethod
    def from_story_string(event_name, parameters, default=None):
        event = Event.resolve_by_type(event_name, default)
        return event._from_story_string(parameters)

    @staticmethod
    def from_parameters(parameters, default=None):
        event_name = parameters.get("event")
        if event_name is not None:
            copied = parameters.copy()
            del copied["event"]
            event = Event.resolve_by_type(event_name, default)
            return event._from_parameters(parameters)
        else:
            return None

    @classmethod
    def _from_story_string(cls, parameters):
        """Called to convert a parsed story line into an event."""
        return cls(parameters.get("timestamp"))

    def as_dict(self):
        return {
            "event": self.type_name,
            "timestamp": self.timestamp,
        }

    @classmethod
    def _from_parameters(cls, parameters):
        """Called to convert a dictionary of parameters to an event.

        By default uses the same implementation as the story line
        conversation ``_from_story_string``. But the subclass might
        decide to handle parameters differently if the parsed parameters
        don't origin from a story file."""

        return cls._from_story_string(parameters)

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

    def __init__(self, text,
                 intent=None,
                 entities=None,
                 parse_data=None,
                 timestamp=None):
        self.text = text
        self.intent = intent if intent else {}
        self.entities = entities if entities else []

        if parse_data:
            self.parse_data = parse_data
        else:
            self.parse_data = {
                "intent": self.intent,
                "entities": self.entities,
                "text": text,
            }

        super(UserUttered, self).__init__(timestamp)

    @staticmethod
    def _from_parse_data(text, parse_data, timestamp=None):
        return UserUttered(text, parse_data["intent"], parse_data["entities"],
                           parse_data,
                           timestamp)

    def __hash__(self):
        return hash((self.text, self.intent.get("name"),
                     jsonpickle.encode(self.entities)))

    def __eq__(self, other):
        if not isinstance(other, UserUttered):
            return False
        else:
            return (self.text, self.intent.get("name"),
                    jsonpickle.encode(self.entities), self.parse_data) == \
                   (other.text, other.intent.get("name"),
                    jsonpickle.encode(other.entities), other.parse_data)

    def __str__(self):
        return ("UserUttered(text: {}, intent: {}, "
                "entities: {})".format(self.text, self.intent, self.entities))

    @staticmethod
    def empty():
        return UserUttered(None)

    def as_dict(self):
        d = super(UserUttered, self).as_dict()
        d.update({
            "text": self.text,
            "parse_data": self.parse_data,
        })
        return d

    @classmethod
    def _from_story_string(cls, parameters):
        try:
            return cls._from_parse_data(parameters.get("text"),
                                        parameters.get("parse_data"),
                                        parameters.get("timestamp"))
        except KeyError as e:
            raise ValueError("Failed to parse bot uttered event. {}".format(e))

    def as_story_string(self):
        if self.intent:
            if self.entities:
                ent_string = json.dumps({ent['entity']: ent['value']
                                         for ent in self.entities})
            else:
                ent_string = ""

            return "{intent}{entities}".format(
                    intent=self.intent.get("name", ""),
                    entities=ent_string)
        else:
            return self.text

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None

        tracker.latest_message = self


# noinspection PyProtectedMember
class BotUttered(Event):
    """The bot has said something to the user.

    This class is not used in the story training as it is contained in the

    ``ActionExecuted`` class. An entry is made in the ``Tracker``."""

    type_name = "bot"

    def __init__(self, text=None, data=None, timestamp=None):
        self.text = text
        self.data = data
        super(BotUttered, self).__init__(timestamp)

    def __hash__(self):
        return hash((self.text, jsonpickle.encode(self.data)))

    def __eq__(self, other):
        if not isinstance(other, BotUttered):
            return False
        else:
            return (self.text, jsonpickle.encode(self.data)) == \
                   (other.text, jsonpickle.encode(other.data))

    def __str__(self):
        return ("BotUttered(text: {}, data: {})"
                "".format(self.text, json.dumps(self.data, indent=2)))

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None

        tracker.latest_bot_utterance = self

    def as_story_string(self):
        return None

    @staticmethod
    def empty():
        return BotUttered()

    def as_dict(self):
        d = super(BotUttered, self).as_dict()
        d.update({
            "text": self.text,
            "data": self.data,
        })
        return d

    @classmethod
    def _from_parameters(cls, parameters):
        try:
            return BotUttered(parameters.get("text"),
                              parameters.get("data"),
                              parameters.get("timestamp"))
        except KeyError as e:
            raise ValueError("Failed to parse bot uttered event. {}".format(e))


# TODO: DEPRECATED - remove in version 0.10.0
# noinspection PyProtectedMember
class TopicSet(Event):
    """The topic of conversation has changed.

    As a side effect self.topic will be pushed on to ``Tracker.topic_stack``."""

    type_name = "topic"

    def __init__(self, topic, timestamp=None):
        self.topic = topic
        super(TopicSet, self).__init__(timestamp)

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
    def _from_story_string(cls, parameters):
        topic = first_key(parameters, default_key="name")

        if topic is not None:
            return TopicSet(topic)
        else:
            return None

    def as_dict(self):
        d = super(TopicSet, self).as_dict()
        d.update({"topic": self.topic})
        return d

    @classmethod
    def _from_parameters(cls, parameters):
        try:
            return TopicSet(parameters.get("topic"),
                            parameters.get("timestamp"))
        except KeyError as e:
            raise ValueError("Failed to parse set topic event. {}".format(e))

    def apply_to(self, tracker):
        pass


# noinspection PyProtectedMember
class SlotSet(Event):
    """The user has specified their preference for the value of a ``slot``.

    As a side effect the ``Tracker``'s slots will be updated so
    that ``tracker.slots[key]=value``."""

    type_name = "slot"

    def __init__(self, key, value=None, timestamp=None):
        self.key = key
        self.value = value
        super(SlotSet, self).__init__(timestamp)

    def __str__(self):
        return "SlotSet(key: {}, value: {})".format(self.key, self.value)

    def __hash__(self):
        return hash((self.key, jsonpickle.encode(self.value)))

    def __eq__(self, other):
        if not isinstance(other, SlotSet):
            return False
        else:
            return (self.key, self.value) == (other.key, other.value)

    def as_story_string(self):
        props = json.dumps({self.key: self.value})
        return "{name}{props}".format(name=self.type_name, props=props)

    @classmethod
    def _from_story_string(cls, parameters):
        slots = []
        for slot_key, slot_val in parameters.items():
            slots.append(SlotSet(slot_key, slot_val))

        if slots:
            return slots
        else:
            return None

    def as_dict(self):
        d = super(SlotSet, self).as_dict()
        d.update({
            "name": self.key,
            "value": self.value,
        })
        return d

    @classmethod
    def _from_parameters(cls, parameters):
        try:
            return SlotSet(parameters.get("name"),
                           parameters.get("value"),
                           parameters.get("timestamp"))
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
        from rasa_core.actions.action import ActionListen
        tracker._reset()
        tracker.follow_up_action = ActionListen()


# noinspection PyProtectedMember
class UserUtteranceReverted(Event):
    """Bot reverts everything until before the most recent user message. 
    
    The bot will revert all events after the latest `UserUttered`, this 
    also means that the last event on the tracker is usually `action_listen` 
    and the bot is waiting for a new user message."""

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
                 kill_on_user_message=True, timestamp=None):
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
        super(ReminderScheduled, self).__init__(timestamp)

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

    def _data_obj(self):
        return {
            "action": self.action_name,
            "date_time": self.trigger_date_time.isoformat(),
            "name": self.name,
            "kill_on_user_msg": self.kill_on_user_message
        }

    def as_story_string(self):
        props = json.dumps(self._data_obj())
        return "{name}{props}".format(name=self.type_name, props=props)

    def as_dict(self):
        d = super(ReminderScheduled, self).as_dict()
        d.update(self._data_obj())
        return d

    @classmethod
    def _parse_trigger_time(self, date_time):
        return datetime.datetime.strptime(date_time[:19], '%Y-%m-%dT%H:%M:%S')

    @classmethod
    def _from_story_string(cls, parameters):
        logger.info("Reminders will be ignored during training, "
                    "which should be ok.")
        trigger_date_time = cls._parse_trigger_time(parameters.get("date_time"))
        return ReminderScheduled(parameters.get("action"),
                                 trigger_date_time,
                                 parameters.get("name", None),
                                 parameters.get("kill_on_user_msg", True),
                                 parameters.get("timestamp"))


# noinspection PyProtectedMember
class ActionReverted(Event):
    """Bot undoes its last action.

    The bot everts everything until before the most recent action.
    This includes the action itself, as well as any events that 
    action created, like set slot events - the bot will now 
    predict a new action using the state before the most recent 
    action."""

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

    def __init__(self, path=None, timestamp=None):
        self.path = path
        super(StoryExported, self).__init__(timestamp)

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
        if self.path:
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

    type_name = "action"

    def __init__(self, action_name, timestamp=None):
        self.action_name = action_name
        self.unpredictable = False
        super(ActionExecuted, self).__init__(timestamp)

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
    def _from_story_string(cls, parameters):
        return ActionExecuted(parameters.get("name"),
                              parameters.get("timestamp"))

    def as_dict(self):
        d = super(ActionExecuted, self).as_dict()
        d.update({"name": self.action_name})
        return d

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None

        tracker.latest_action_name = self.action_name


class AgentUttered(Event):
    """The agent has said something to the user.

    This class is not used in the story training as it is contained in the
    ``ActionExecuted`` class. An entry is made in the ``Tracker``."""

    type_name = "agent"

    def __init__(self, text=None, data=None, timestamp=None):
        self.text = text
        self.data = data
        super(AgentUttered, self).__init__(timestamp)

    def __hash__(self):
        return hash((self.text, jsonpickle.encode(self.data)))

    def __eq__(self, other):
        if not isinstance(other, AgentUttered):
            return False
        else:
            return (self.text, jsonpickle.encode(self.data)) == \
                   (other.text, jsonpickle.encode(other.data))

    def __str__(self):
        return "AgentUttered(text: {}, data: {})".format(
                self.text, json.dumps(self.data, indent=2))

    def apply_to(self, tracker):
        # type: (DialogueStateTracker) -> None

        pass

    def as_story_string(self):
        return None

    def as_dict(self):
        d = super(AgentUttered, self).as_dict()
        d.update({
            "text": self.text,
            "data": self.data,
        })
        return d

    @staticmethod
    def empty():
        return AgentUttered()

    @classmethod
    def _from_parameters(cls, parameters):
        try:
            return AgentUttered(parameters.get("text"),
                                parameters.get("data"),
                                parameters.get("timestamp"))
        except KeyError as e:
            raise ValueError("Failed to parse agent uttered event. "
                             "{}".format(e))
