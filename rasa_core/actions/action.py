from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import typing
from typing import List, Text

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.dispatcher import Dispatcher
    from rasa_core.events import Event
    from rasa_core.domain import Domain

logger = logging.getLogger(__name__)

ACTION_LISTEN_NAME = "action_listen"

ACTION_RESTART_NAME = "action_restart"

ACTION_DEFAULT_FALLBACK_NAME = "action_default_fallback"


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
            tracker (DialogueStateTracker): the state tracker for the current user.
                You can access slot values using ``tracker.get_slot(slot_name)``
                and the most recent user message is ``tracker.latest_message.text``.
            dispatcher (Dispatcher): the dispatcher which is used to send messages back
                to the user. Use ``dipatcher.utter_message()`` or any other :class:`Dispatcher` method.
            domain (Domain): the bot's domain

        Returns:
            List: A list of :class:`Event` instances

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

        if domain.random_template_for("utter_default") is not None:
            dispatcher.utter_template("utter_default", tracker,
                                      silent_fail=True)

        return [UserUtteranceReverted()]
