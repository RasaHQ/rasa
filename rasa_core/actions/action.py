from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from collections import namedtuple

import requests
import typing
from requests.auth import HTTPBasicAuth
from typing import List, Text

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.dispatcher import Dispatcher
    from rasa_core.events import Event

logger = logging.getLogger(__name__)

ACTION_LISTEN_NAME = "action_listen"

ACTION_RESTART_NAME = "action_restart"


def ensure_action_name_uniqueness(actions):
    actual_action_names = set()  # used to collect unique action names
    for a in actions:
        if a.name() in actual_action_names:
            raise ValueError(
                    "Action names are not unique! Found two actions with name"
                    " '{}'. Either rename or remove one of them."
                    "".format(a.name()))
        else:
            actual_action_names.add(a.name())


def actions_from_names(action_names, action_endpoint):
    # type: (List[Text], ActionEndpointConfig) -> List[Action]
    """Converts the names of actions into class instances."""

    actions = []
    for name in action_names:
        if name.startswith("utter_"):
            actions.append(UtterAction(name))
        else:
            actions.append(RemoteAction(name, action_endpoint))

    return actions


class Action(object):
    """Next action to be taken in response to a dialogue state."""

    def resets_topic(self):
        # type: () -> bool
        """Indicator if this action resets the topic when run."""

        return False

    def name(self):
        # type: () -> Text
        """Unique identifier of this simple action."""

        raise NotImplementedError

    def run(self, dispatcher, tracker, domain):
        # type: (Dispatcher, DialogueStateTracker, Domain) -> List[Event]
        """Execute the side effects of this action.

        Return a list of events (i.e. instructions to update tracker state)

        :param tracker: user state tracker
        :param dispatcher: communication channel
        """

        raise NotImplementedError

    def __str__(self):
        return "Action('{}')".format(self.name())


class UtterAction(Action):
    """An action which only effect is to utter a template when it is run.

    Both, name and utter template, need to be specified using
    the `name` method."""

    def __init__(self, name):
        self._name = name

    def run(self, dispatcher, tracker, domain):
        """Simple run implementation uttering a (hopefully defined) template."""

        dispatcher.utter_template(self.name(),
                                  filled_slots=tracker.current_slot_values())
        return []

    def name(self):
        return self._name

    def __str__(self):
        return "UtterAction('{}')".format(self.name())


class ActionListen(Action):
    """The first action in any turn - bot waits for a user message.

    The bot should stop taking further actions and wait for the user to say
    something."""

    def name(self):
        return ACTION_LISTEN_NAME

    def run(self, dispatcher, tracker, domain):
        return []


class ActionRestart(Action):
    """Resets the tracker to its initial state.

    Utters the restart template if available."""

    def name(self):
        return ACTION_RESTART_NAME

    def run(self, dispatcher, tracker, domain):
        from rasa_core.events import Restarted

        # only utter the template if it is available
        dispatcher.utter_template("utter_restart", silent_fail=True)
        return [Restarted()]


ActionEndpointConfig = namedtuple('ActionEndpointConfig', ["url",
                                                           "headers",
                                                           "basic_auth"])


class RemoteAction(Action):
    def __init__(self, name, action_endpoint):
        # type: (Text, ActionEndpointConfig) -> None

        self._name = name
        self.action_endpoint = action_endpoint

    def _action_call_format(self, tracker):
        tracker_state = tracker.current_state(
                should_include_events=True,
                only_events_after_latest_restart=True)

        return {
            "next_action": self._name,
            "tracker": tracker_state
        }

    def _validate_action_result(self, result):
        return True

    def run(self, dispatcher, tracker, domain):
        json = self._action_call_format(tracker)

        if self.action_endpoint.headers:
            headers = self.action_endpoint.headers.copy()
        else:
            headers = {}
        headers["Content-Type"] = "application/json"

        if self.action_endpoint.basic_auth:
            auth = HTTPBasicAuth(self.action_endpoint.basic_auth["username"],
                                 self.action_endpoint.basic_auth["password"])
        else:
            auth = None

        response = requests.post(self.action_endpoint.url,
                                 headers=headers,
                                 auth=auth,
                                 json=json)

        response.raise_for_status()

        self._validate_action_result(response.json())

        return []

    def name(self):
        return self._name
