from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import importlib
import inspect
import logging
import pkgutil

from builtins import str
from rasa_core.domain import TemplateDomain
from typing import Text

from rasa_core import utils
from rasa_core.actions import Action
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class CollectingDispatcher(object):
    """Send messages back to user"""

    def __init__(self):
        # type: () -> None

        self.messages = []

    def utter_custom_message(self, *elements):
        # type: (*Dict[Text, Any]) -> None
        """Sends a message with custom elements to the output channel."""

        message = {"text": None, "elements": elements}

        self.messages.append(message)

    def utter_button_message(self, text, buttons, **kwargs):
        # type: (Text, List[Dict[Text, Any]], **Any) -> None
        """Sends a message with buttons to the output channel."""

        message = {"text": text, "buttons": buttons}
        message.update(kwargs)

        self.messages.append(message)

    def utter_attachment(self, attachment):
        # type: (Text) -> None
        """Send a message to the client with attachments."""

        message = {"text": None, "attachement": attachment}

        self.messages.append(message)

    # TODO: TB - deprecate this function?
    def utter_button_template(self, template, buttons, tracker,
                              silent_fail=False,
                              **kwargs):
        # type: (Text, List[Dict[Text, Any]], **Any) -> None
        """Sends a message template with buttons to the output channel."""

        message = {"template": template, "buttons": buttons}
        message.update(kwargs)

        self.messages.append(message)

    def utter_template(self,
                       template,  # type: Text
                       tracker,
                       silent_fail=False,  # type: bool
                       **kwargs  # type: ** Any
                       ):
        # type: (...) -> None
        """"Send a message to the client based on a template."""

        message = {"template": template}
        message.update(kwargs)

        self.messages.append(message)


class ActionExecutor(object):
    def __init__(self):
        self.actions = {}

    def register_action(self, action):
        if inspect.isclass(action):
            if action.__module__.startswith("rasa_core."):
                logger.warning("Skipping built in Action {}.".format(action))
                return
            else:
                action = action()
        if isinstance(action, Action):
            self.register_function(action.name(), action.run)
        else:
            raise Exception("You can only register instances or subclasses of "
                            "type Action. If you want to directly register "
                            "a function, use `register_function` instead.")

    def register_function(self, name, f):
        logger.debug("Registered function for '{}'.".format(name))
        valid_keys = utils.arguments_of(f)
        if len(valid_keys) < 3:
            raise Exception("You can only register functions that take "
                            "3 parameters as arguments. The three parameters "
                            "your function will receive are: dispatcher, "
                            "tracker, domain. Your function accepts only {} "
                            "parameters.".format(len(valid_keys)))
        self.actions[name] = f

    def _import_submodules(self, package, recursive=True):
        """ Import all submodules of a module, recursively, including
        subpackages

        :param package: package (name or actual module)
        :type package: str | module
        :rtype: dict[str, types.ModuleType]
        """
        if isinstance(package, str):
            package = importlib.import_module(package)
        if not getattr(package, '__path__', None):
            return

        results = {}
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_name = package.__name__ + '.' + name
            results[full_name] = importlib.import_module(full_name)
            if recursive and is_pkg:
                self._import_submodules(full_name)

    def register_package(self, package):

        self._import_submodules(package)

        actions = utils.all_subclasses(Action)

        for action in actions:
            if not action.__module__.startswith("rasa_core."):
                self.register_action(action)

    def _create_api_response(self, events, messages):
        return{
            "events": [e.as_dict() for e in events] if events else [],
            "responses": messages
        }

    def run(self, action_call):
        action_name = action_call.get("next_action")
        if action_name:
            action = self.actions.get(action_name)
            if not action:
                raise Exception("No registered Action found for name '{}'."
                                "".format(action_name))
            sender_id = action_call.get("sender_id")
            tracker_json = action_call.get("tracker")

            domain = TemplateDomain.from_dict(action_call.get("domain", {}))
            tracker = DialogueStateTracker.from_dict(sender_id,
                                                     tracker_json.get("events"),
                                                     domain.slots)
            dispatcher = CollectingDispatcher()

            events = action(dispatcher, tracker, domain)

            return self._create_api_response(events, dispatcher.messages)
        else:
            logger.warning("Received an action call without an action.")
