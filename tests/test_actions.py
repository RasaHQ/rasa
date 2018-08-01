from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from rasa_core.actions import action
from rasa_core.actions.action import (
    ActionRestart, UtterAction,
    ActionListen, RemoteAction)
from rasa_core.domain import TemplateDomain
from rasa_core.events import Restarted
from rasa_core.trackers import DialogueStateTracker


def test_restart(default_dispatcher_collecting, default_domain):
    tracker = DialogueStateTracker("default",
                                   default_domain.slots)
    events = ActionRestart().run(default_dispatcher_collecting, tracker,
                                 default_domain)
    assert events == [Restarted()]


def test_text_format():
    assert "{}".format(ActionListen()) == \
           "Action('action_listen')"
    assert "{}".format(UtterAction("my_action_name")) == \
           "UtterAction('my_action_name')"


def test_action_factory_module_import():
    instantiated_actions = action.actions_from_names(
            ["random_name", "utter_test"], None)
    assert len(instantiated_actions) == 2
    assert isinstance(instantiated_actions[0], RemoteAction)
    assert instantiated_actions[0].name() == "random_name"

    assert isinstance(instantiated_actions[1], UtterAction)
    assert instantiated_actions[1].name() == "utter_test"


def test_domain_action_instantiation():
    instantiated_actions = TemplateDomain.instantiate_actions(
            ["my_module.ActionTest", "utter_test"], None)
    assert len(instantiated_actions) == 5
    assert instantiated_actions[0].name() == "action_listen"
    assert instantiated_actions[1].name() == "action_restart"
    assert instantiated_actions[2].name() == "action_default_fallback"
    assert instantiated_actions[3].name() == "my_module.ActionTest"
    assert instantiated_actions[4].name() == "utter_test"


def test_action_factory_fails_on_duplicated_actions():
    with pytest.raises(ValueError):
        TemplateDomain.instantiate_actions(
                ["action_listen", "random_name", "random_name"], None)


def test_action_factory_fails_on_duplicated_builtin_actions():
    actions = ["action_listen",
               "action_listen",
               "utter_test"]
    with pytest.raises(ValueError):
        TemplateDomain.instantiate_actions(actions, None)
