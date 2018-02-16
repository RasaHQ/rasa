from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import re

from typing import Text, List

from rasa_core import utils
from rasa_core.actions import Action
from rasa_core.actions.action import UtterAction

logger = logging.getLogger(__name__)


def action_factory_by_name(name):
    if name == "local" or name is None:  # this is the default factory
        return local_action_factory
    elif name == "remote":
        return remote_action_factory
    else:
        raise Exception("Unknown executor name '{}'.".format(name))


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


def local_action_factory(action_classes, action_names, utter_templates):
    # type: (List[Text], List[Text], List[Text]) -> List[Action]
    """Converts the names of actions into class instances."""

    def _action_class(action_name):
        # type: (Text) -> Action
        """Tries to create an instance by importing and calling the class."""

        try:
            cls = utils.class_from_module_path(action_name)
            return cls()
        except ImportError as e:
            if len(e.args) > 0:
                erx = re.compile("No module named '?(.*?)'?$")
                matched = erx.search(e.args[0])
                if matched and matched.group(1) in action_name:
                    # we only want to capture exceptions that are raised by the
                    # class itself, not by other packages that fail to import
                    raise ValueError(
                        "Action '{}' doesn't correspond to a template / "
                        "action. Remember to prefix actions that should "
                        "utter a template with `utter_`. "
                        "Error: {}".format(action_name, e))
            # raises the original exception again
            raise
        except (AttributeError, KeyError) as e:
            raise ValueError(
                    "Action '{}' doesn't correspond to a template / action. "
                    "Module doesn't contain a class with this name. "
                    "Remember to prefix actions that should utter a template "
                    "with `utter_`. Error: {}".format(action_name, e))

    actions = []

    for name in action_classes:
        if name in utter_templates:
            actions.append(UtterAction(name))
        else:
            actions.append(_action_class(name))

    return actions


def remote_action_factory(action_classes, action_names, utter_templates):
    # type: (List[Text], List[Text], List[Text]) -> List[Action]
    """Converts the names of actions into class instances."""

    if action_names:
        remote_action_ids = action_names
    else:
        # if we do not have action names - we use the class names as identifiers
        # for the remote actions
        remote_action_ids = action_classes

    return [RemoteAction(aid) for aid in remote_action_ids]


class RemoteAction(Action):
    def __init__(self, name):
        self._name = name

    def run(self, dispatcher, tracker, domain):
        logger.info("Remote action can not be run locally.")
        return []

    def name(self):
        return self._name
