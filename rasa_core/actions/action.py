from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import requests
import typing
from requests.auth import HTTPBasicAuth
from typing import List, Text

from rasa_core import events, utils

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
    # type: (List[Text], EndpointConfig) -> List[Action]
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
                                  tracker)
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
        dispatcher.utter_template("utter_restart", tracker, silent_fail=True)
        return [Restarted()]


class EndpointConfig(object):
    def __init__(self, url, params=None, headers=None, basic_auth=None):
        self.url = url
        self.params = params if params else {}
        self.headers = headers if headers else {}
        self.basic_auth = basic_auth

    @classmethod
    def from_dict(cls, data):
        return EndpointConfig(
                data.get("url"),
                data.get("params"),
                data.get("headers"),
                data.get("basic_auth"))


class RemoteAction(Action):
    def __init__(self, name, action_endpoint):
        # type: (Text, EndpointConfig) -> None

        self._name = name
        self.action_endpoint = action_endpoint

    def _action_call_format(self, tracker, domain):
        tracker_state = tracker.current_state(
                should_include_events=True,
                only_events_after_latest_restart=True)

        return {
            "next_action": self._name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "domain": domain.as_dict()
        }

    def _validate_action_result(self, result):
        # TODO: TB - make sure the json is valid
        return True

    def _validate_events(self, evts):
        # TODO: TB - make sure no invalid events are logged
        # e.g. setting of non existent slots
        return True

    def _handle_responses(self, responses, dispatcher, tracker):
        for response in responses:
            if "template" in response:
                draft = dispatcher.nlg.generate(
                        response["template"],
                        tracker,
                        dispatcher.output_channel.name())
                del response["template"]
            else:
                draft = {}

            if "buttons" in response:
                if "buttons" not in draft:
                    draft["buttons"] = []
                draft["buttons"].extend(response["buttons"])
                del response["buttons"]

            draft.update(response)
            dispatcher.utter_response(draft)

    def run(self, dispatcher, tracker, domain):
        json = self._action_call_format(tracker, domain)

        response = utils.post_json_to_endpoint(json, self.action_endpoint)
        response.raise_for_status()

        response_data = response.json()

        self._validate_action_result(response_data)

        events_json = response_data.get("events", [])
        responses = response_data.get("responses", [])

        self._handle_responses(responses, dispatcher, tracker)

        evts = events.deserialise_events(events_json)

        self._validate_events(evts)

        return evts

    def name(self):
        return self._name
