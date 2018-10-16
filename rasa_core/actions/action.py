from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import requests
import typing
from typing import List, Text, Optional, Dict, Any

from rasa_core import events
from rasa_core.constants import DOCS_BASE_URL, DEFAULT_REQUEST_TIMEOUT
from rasa_core.utils import EndpointConfig

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.dispatcher import Dispatcher
    from rasa_core.events import Event
    from rasa_core.domain import Domain

logger = logging.getLogger(__name__)

ACTION_LISTEN_NAME = "action_listen"

ACTION_RESTART_NAME = "action_restart"

ACTION_DEFAULT_FALLBACK_NAME = "action_default_fallback"


def default_actions():
    # type: () -> List[Action]
    """List default actions."""
    return [ActionListen(), ActionRestart(), ActionDefaultFallback()]


def default_action_names():
    # type: () -> List[Text]
    """List default action names."""
    return [a.name() for a in default_actions()]


def combine_user_with_default_actions(user_actions):
    # remove all user actions that overwrite default actions
    # this logic is a bit reversed, you'd think that we should remove
    # the action name from the default action names if the user overwrites
    # the action, but there are some locations in the code where we
    # implicitly assume that e.g. "action_listen" is always at location
    # 0 in this array. to keep it that way, we remove the duplicate
    # action names from the users list instead of the defaults
    unique_user_actions = [a
                           for a in user_actions
                           if a not in default_action_names()]
    return default_action_names() + unique_user_actions


def ensure_action_name_uniqueness(action_names):
    # type: (List[Text]) -> None
    """Check and raise an exception if there are two actions with same name."""

    unique_action_names = set()  # used to collect unique action names
    for a in action_names:
        if a in unique_action_names:
            raise ValueError(
                    "Action names are not unique! Found two actions with name"
                    " '{}'. Either rename or remove one of them.".format(a))
        else:
            unique_action_names.add(a)


def action_from_name(name, action_endpoint, user_actions):
    # type: (Text, Optional[EndpointConfig], List[Text]) -> Action
    """Return an action instance for the name."""

    defaults = {a.name(): a for a in default_actions()}

    if name in defaults and name not in user_actions:
        return defaults.get(name)
    elif name.startswith("utter_"):
        return UtterAction(name)
    else:
        return RemoteAction(name, action_endpoint)


def actions_from_names(action_names, action_endpoint, user_actions):
    # type: (List[Text], Optional[EndpointConfig], List[Text]) -> List[Action]
    """Converts the names of actions into class instances."""

    return [action_from_name(name, action_endpoint, user_actions)
            for name in action_names]


class Action(object):
    """Next action to be taken in response to a dialogue state."""

    def name(self):
        # type: () -> Text
        """Unique identifier of this simple action."""

        raise NotImplementedError

    def run(self, dispatcher, tracker, domain):
        # type: (Dispatcher, DialogueStateTracker, Domain) -> List[Event]
        """
        Execute the side effects of this action.

        Args:
            dispatcher (Dispatcher): the dispatcher which is used to send
                messages back to the user. Use ``dispatcher.utter_message()``
                or any other :class:`rasa_core.dispatcher.Dispatcher` method.
            tracker (DialogueStateTracker): the state tracker for the current
                user. You can access slot values using
                ``tracker.get_slot(slot_name)`` and the most recent user
                message is ``tracker.latest_message.text``.
            domain (Domain): the bot's domain

        Returns:
            List[Event]: A list of :class:`rasa_core.events.Event` instances
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
        dispatcher.utter_template("utter_restart", tracker,
                                  silent_fail=True)
        return [Restarted()]


class ActionDefaultFallback(Action):
    """Executes the fallback action and goes back to the previous state
    of the dialogue"""

    def name(self):
        return ACTION_DEFAULT_FALLBACK_NAME

    def run(self, dispatcher, tracker, domain):
        from rasa_core.events import UserUtteranceReverted

        dispatcher.utter_template("utter_default", tracker,
                                  silent_fail=True)

        return [UserUtteranceReverted()]


class RemoteAction(Action):
    def __init__(self, name, action_endpoint):
        # type: (Text, Optional[EndpointConfig]) -> None

        self._name = name
        self.action_endpoint = action_endpoint

    def _action_call_format(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> Dict[Text, Any]
        """Create the request json send to the action server."""
        from rasa_core.trackers import EventVerbosity

        tracker_state = tracker.current_state(EventVerbosity.ALL)

        return {
            "next_action": self._name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "domain": domain.as_dict()
        }

    @staticmethod
    def action_response_format_spec():
        """Expected response schema for an Action endpoint.

        Used for validation of the response returned from the
        Action endpoint."""
        return {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event": {"type": "string"}
                        }
                    }

                },
                "responses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                    }
                }
            },
        }

    def _validate_action_result(self, result):
        from jsonschema import validate
        from jsonschema import ValidationError

        try:
            validate(result, self.action_response_format_spec())
            return True
        except ValidationError as e:
            e.message += (
                ". Failed to validate Action server response from API, "
                "make sure your response from the Action endpoint is valid. "
                "For more information about the format visit "
                "{}/customactions/".format(DOCS_BASE_URL))
            raise e

    @staticmethod
    def _utter_responses(responses,  # type: List[Dict[Text, Any]]
                         dispatcher,  # type: Dispatcher
                         tracker  # type: DialogueStateTracker
                         ):
        # type: (...) -> None
        """Use the responses generated by the action endpoint and utter them.

        Uses the normal dispatcher to utter the responses from the action
        endpoint."""

        for response in responses:
            if "template" in response:
                kwargs = response.copy()
                del kwargs["template"]
                draft = dispatcher.nlg.generate(
                        response["template"],
                        tracker,
                        dispatcher.output_channel.name(),
                        **kwargs)
                if not draft:
                    continue

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

        if not self.action_endpoint:
            raise Exception("The model predicted the custom action '{}' "
                            "but you didn't configure an endpoint to "
                            "run this custom action. Please take a look at "
                            "the docs and set an endpoint configuration. "
                            "{}/customactions/"
                            "".format(self.name(), DOCS_BASE_URL))

        try:
            logger.debug("Calling action endpoint to run action '{}'."
                         "".format(self.name()))
            response = self.action_endpoint.request(
                    json=json, method="post", timeout=DEFAULT_REQUEST_TIMEOUT)
            response.raise_for_status()
            response_data = response.json()

            self._validate_action_result(response_data)
        except requests.exceptions.ConnectionError as e:

            logger.error("Failed to run custom action '{}'. Couldn't connect "
                         "to the server at '{}'. Is the server running? "
                         "Error: {}".format(self.name(),
                                            self.action_endpoint.url,
                                            e))
            raise Exception("Failed to execute custom action.")
        except requests.exceptions.HTTPError as e:

            logger.error("Failed to run custom action '{}'. Action server "
                         "responded with a non 200 status code of {}. "
                         "Make sure your action server properly runs actions "
                         "and returns a 200 once the action is executed. "
                         "Error: {}".format(self.name(),
                                            e.response.status_code,
                                            e))
            raise Exception("Failed to execute custom action.")

        events_json = response_data.get("events", [])
        responses = response_data.get("responses", [])

        self._utter_responses(responses, dispatcher, tracker)

        evts = events.deserialise_events(events_json)

        return evts

    def name(self):
        return self._name
