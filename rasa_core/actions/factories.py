from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.actions.action import UtterAction
from typing import Text, Optional, List

from rasa_core.actions import Action
from rasa_core import utils

logger = logging.getLogger(__name__)


def action_factory_by_name(name):
    if name == "local" or name is None:     # this is the default factory
        return local_action_factory
    elif name == "remote":
        return remote_action_factory
    else:
        raise Exception("Unknown executor name '{}'.".format(name))


def local_action_factory(action_names, utter_templates):
    # type: (List[Text]) -> List[Action]
    """Converts the names of actions into class instances."""
    from rasa_core.domain import Domain

    def _action_class(action_name):
        # type: (Text) -> Action
        """Tries to create an instance by importing and calling the class."""

        try:
            cls = utils.class_from_module_path(action_name)
            return cls()
        except ImportError as e:
            raise ValueError(
                    "Action '{}' doesn't correspond to a template / action. "
                    "Remember to prefix actions that should utter a template "
                    "with `utter_`. Error: {}".format(action_name, e))
        except (AttributeError, KeyError) as e:
            raise ValueError(
                    "Action '{}' doesn't correspond to a template / action. "
                    "Module doesn't contain a class with this name. "
                    "Remember to prefix actions that should utter a template "
                    "with `utter_`. Error: {}".format(action_name, e))

    default_actions = {a.name(): a for a in Domain.DEFAULT_ACTIONS}
    actions = []

    for name in action_names:
        if name in default_actions:
            actions.append(default_actions[name])
        elif name in utter_templates:
            actions.append(UtterAction(name))
        else:
            actions.append(_action_class(name))

    # TODO: double check that action names are unique?
    return actions


def remote_action_factory(action_names, utter_templates):
    # type: (List[Text]) -> List[Action]
    """Converts the names of actions into class instances."""
    from rasa_core.domain import Domain

    default_actions = {a.name(): a for a in Domain.DEFAULT_ACTIONS}
    actions = []

    for name in action_names:
        if name in default_actions:
            actions.append(default_actions[name])
        else:
            actions.append(RemoteAction(name))

    return actions


class RemoteAction(Action):
    def __init__(self, name):
        self._name = name

    def run(self, dispatcher, tracker, domain):
        logger.info("Remote action can not be run locally.")
        return []

    def name(self):
        return self._name
