from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from rasa_core.actions.factories import action_factory_by_name, RemoteAction

from rasa_core.actions.action import (
    ActionRestart, UtterAction,
    ActionListen, Action)
from rasa_core.domain import TemplateDomain
from rasa_core.events import Restarted
from rasa_core.trackers import DialogueStateTracker


def test_restart(default_dispatcher_cmd, default_domain):
    tracker = DialogueStateTracker("default",
                                   default_domain.slots)
    events = ActionRestart().run(default_dispatcher_cmd, tracker,
                                 default_domain)
    assert events == [Restarted()]


def test_text_format():
    assert "{}".format(ActionListen()) == \
           "Action('action_listen')"
    assert "{}".format(UtterAction("my_action_name")) == \
           "UtterAction('my_action_name')"


def test_action_factories():
    assert action_factory_by_name("local") is not None
    assert action_factory_by_name("remote") is not None
    with pytest.raises(Exception):
        action_factory_by_name("unknown_name")


def test_local_action_factory_module_import():
    instantiated_actions = action_factory_by_name("local")(
            ["rasa_core.actions.action.ActionRestart",
             "utter_test"], None, ["utter_test"])
    assert len(instantiated_actions) == 2
    assert isinstance(instantiated_actions[0], ActionRestart)
    assert isinstance(instantiated_actions[1], UtterAction)


def test_remote_action_factory_module_import():
    instantiated_actions = action_factory_by_name("remote")(
            ["random_name", "utter_test"], None,
            ["utter_test"])
    assert len(instantiated_actions) == 2
    assert isinstance(instantiated_actions[0], RemoteAction)
    assert isinstance(instantiated_actions[1], RemoteAction)


def test_remote_action_factory_preferes_action_names():
    instantiated_actions = action_factory_by_name("remote")(
            ["my_module.ActionTest", "utter_test"],
            ["action_test", "utter_test"],
            ["utter_test"])
    assert len(instantiated_actions) == 2
    assert instantiated_actions[0].name() == "action_test"
    assert instantiated_actions[1].name() == "utter_test"


def test_domain_action_instantiation():
    instantiated_actions = TemplateDomain.instantiate_actions(
            "remote",
            ["my_module.ActionTest", "utter_test"],
            ["action_test", "utter_test"],
            ["utter_test"])
    assert len(instantiated_actions) == 5
    assert instantiated_actions[0].name() == "action_listen"
    assert instantiated_actions[1].name() == "action_restart"
    assert instantiated_actions[2].name() == "action_default_fallback"
    assert instantiated_actions[3].name() == "action_test"
    assert instantiated_actions[4].name() == "utter_test"


def test_local_action_factory_module_import_fails_on_invalid():
    with pytest.raises(ValueError):
        action_factory_by_name("local")(["random_name"], None, ["utter_test"])

    with pytest.raises(ValueError):
        action_factory_by_name("local")(["utter_default"], None, ["utter_test"])

    with pytest.raises(ValueError):
        action_factory_by_name("local")(["examples.UnavailableClass"],
                                        None,
                                        ["utter_test"])

    with pytest.raises(ValueError):
        action_factory_by_name("local")(["nonexistant.UnavailableClass"],
                                        None,
                                        ["utter_test"])


def test_remote_action_factory_fails_on_duplicated_actions():
    with pytest.raises(ValueError):
        TemplateDomain.instantiate_actions(
                "remote", ["action_listen", "random_name", "random_name"],
                None, ["utter_test"])


def test_local_action_factory_fails_on_duplicated_actions():
    actions = ["action_listen",
               "rasa_core.actions.action.ActionListen",
               "utter_test"]
    with pytest.raises(ValueError):
        TemplateDomain.instantiate_actions(
                "local", actions, None, ["utter_test"])
