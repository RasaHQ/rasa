from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from rasa_core.actions.factories import action_factory_by_name, RemoteAction

from rasa_core.actions.action import ActionRestart, UtterAction, \
    ActionListen, Action
from rasa_core.events import Restarted
from rasa_core.trackers import DialogueStateTracker


def test_restart(default_dispatcher, default_domain):
    tracker = DialogueStateTracker("default",
                                   default_domain.slots,
                                   default_domain.topics,
                                   default_domain.default_topic)
    events = ActionRestart().run(default_dispatcher, tracker, default_domain)
    assert events == [Restarted()]


def test_text_format():
    assert "{}".format(ActionListen()) == \
           "Action('action_listen')"
    assert "{}".format(UtterAction("my_action_name")) == \
           "UtterAction('my_action_name')"


def test_default_reset_topic():
    assert not Action().resets_topic()


def test_action_factories():
    assert action_factory_by_name("local") is not None
    assert action_factory_by_name("remote") is not None
    with pytest.raises(Exception):
        action_factory_by_name("unknown_name")


def test_local_action_factory_module_import():
    instantiated_actions = action_factory_by_name("local")(
            ["action_listen", "rasa_core.actions.action.ActionListen",
             "utter_test"], ["utter_test"])
    assert len(instantiated_actions) == 3
    assert isinstance(instantiated_actions[0], ActionListen)
    assert isinstance(instantiated_actions[1], ActionListen)
    assert isinstance(instantiated_actions[2], UtterAction)


def test_remote_action_factory_module_import():
    instantiated_actions = action_factory_by_name("remote")(
            ["action_listen", "random_name", "utter_test"], ["utter_test"])
    assert len(instantiated_actions) == 3
    assert isinstance(instantiated_actions[0], ActionListen)
    assert isinstance(instantiated_actions[1], RemoteAction)
    assert isinstance(instantiated_actions[2], RemoteAction)


def test_local_action_factory_module_import_fails_on_invalid():
    with pytest.raises(ValueError):
        action_factory_by_name("local")(["random_name"], ["utter_test"])

    with pytest.raises(ValueError):
        action_factory_by_name("local")(["utter_default"], ["utter_test"])

    with pytest.raises(ValueError):
        action_factory_by_name("local")(["examples.UnavailableClass"],
                                        ["utter_test"])

    with pytest.raises(ValueError):
        action_factory_by_name("local")(["nonexistant.UnavailableClass"],
                                        ["utter_test"])
