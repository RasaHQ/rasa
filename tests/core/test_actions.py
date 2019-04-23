import pytest
from aioresponses import aioresponses

import rasa.core
from rasa.core.actions import action
from rasa.core.actions.action import (
    ACTION_DEACTIVATE_FORM_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ActionExecutionRejection,
    ActionListen,
    ActionRestart,
    RemoteAction,
    UtterAction,
    ACTION_BACK_NAME,
)
from rasa.core.domain import Domain
from rasa.core.events import Restarted, SlotSet, UserUtteranceReverted
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import ClientResponseError, EndpointConfig
from tests.utilities import json_of_latest_request, latest_request


async def test_restart(default_dispatcher_collecting, default_domain):
    tracker = DialogueStateTracker("default", default_domain.slots)
    events = await ActionRestart().run(
        default_dispatcher_collecting, tracker, default_domain
    )
    assert events == [Restarted()]


def test_text_format():
    assert "{}".format(ActionListen()) == "Action('action_listen')"
    assert "{}".format(UtterAction("my_action_name")) == "UtterAction('my_action_name')"


def test_action_instantiation_from_names():
    instantiated_actions = action.actions_from_names(
        ["random_name", "utter_test"], None, ["random_name", "utter_test"]
    )
    assert len(instantiated_actions) == 2
    assert isinstance(instantiated_actions[0], RemoteAction)
    assert instantiated_actions[0].name() == "random_name"

    assert isinstance(instantiated_actions[1], UtterAction)
    assert instantiated_actions[1].name() == "utter_test"


def test_domain_action_instantiation():
    domain = Domain(
        intent_properties={},
        entities=[],
        slots=[],
        templates={},
        action_names=["my_module.ActionTest", "utter_test"],
        form_names=[],
    )

    instantiated_actions = domain.actions(None)

    assert len(instantiated_actions) == 10
    assert instantiated_actions[0].name() == ACTION_LISTEN_NAME
    assert instantiated_actions[1].name() == ACTION_RESTART_NAME
    assert instantiated_actions[2].name() == ACTION_DEFAULT_FALLBACK_NAME
    assert instantiated_actions[3].name() == ACTION_DEACTIVATE_FORM_NAME
    assert instantiated_actions[4].name() == ACTION_REVERT_FALLBACK_EVENTS_NAME
    assert instantiated_actions[5].name() == (ACTION_DEFAULT_ASK_AFFIRMATION_NAME)
    assert instantiated_actions[6].name() == (ACTION_DEFAULT_ASK_REPHRASE_NAME)
    assert instantiated_actions[7].name() == ACTION_BACK_NAME
    assert instantiated_actions[8].name() == "my_module.ActionTest"
    assert instantiated_actions[9].name() == "utter_test"


def test_domain_fails_on_duplicated_actions():
    with pytest.raises(ValueError):
        Domain(
            intent_properties={},
            entities=[],
            slots=[],
            templates={},
            action_names=["random_name", "random_name"],
            form_names=[],
        )


async def test_remote_action_runs(default_dispatcher_collecting, default_domain):
    tracker = DialogueStateTracker("default", default_domain.slots)

    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload={"events": [], "responses": []},
        )

        await remote_action.run(default_dispatcher_collecting, tracker, default_domain)

        r = latest_request(mocked, "post", "https://example.com/webhooks/actions")

        assert r

        assert json_of_latest_request(r) == {
            "domain": default_domain.as_dict(),
            "next_action": "my_action",
            "sender_id": "default",
            "version": rasa.__version__,
            "tracker": {
                "latest_message": {"entities": [], "intent": {}, "text": None},
                "active_form": {},
                "latest_action_name": None,
                "sender_id": "default",
                "paused": False,
                "latest_event_time": None,
                "followup_action": "action_listen",
                "slots": {"name": None},
                "events": [],
                "latest_input_channel": None,
            },
        }


async def test_remote_action_logs_events(default_dispatcher_collecting, default_domain):
    tracker = DialogueStateTracker("default", default_domain.slots)

    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    response = {
        "events": [{"event": "slot", "value": "rasa", "name": "name"}],
        "responses": [
            {"text": "test text", "buttons": [{"title": "cheap", "payload": "cheap"}]},
            {"template": "utter_greet"},
        ],
    }

    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", payload=response)

        events = await remote_action.run(
            default_dispatcher_collecting, tracker, default_domain
        )

        r = latest_request(mocked, "post", "https://example.com/webhooks/actions")
        assert r

        assert json_of_latest_request(r) == {
            "domain": default_domain.as_dict(),
            "next_action": "my_action",
            "sender_id": "default",
            "version": rasa.__version__,
            "tracker": {
                "latest_message": {"entities": [], "intent": {}, "text": None},
                "active_form": {},
                "latest_action_name": None,
                "sender_id": "default",
                "paused": False,
                "followup_action": "action_listen",
                "latest_event_time": None,
                "slots": {"name": None},
                "events": [],
                "latest_input_channel": None,
            },
        }

    assert events == [SlotSet("name", "rasa")]

    channel = default_dispatcher_collecting.output_channel
    assert channel.messages == [
        {
            "text": "test text",
            "recipient_id": "my-sender",
            "buttons": [{"title": "cheap", "payload": "cheap"}],
        },
        {"text": "hey there None!", "recipient_id": "my-sender"},
    ]


async def test_remote_action_without_endpoint(
    default_dispatcher_collecting, default_domain
):
    tracker = DialogueStateTracker("default", default_domain.slots)

    remote_action = action.RemoteAction("my_action", None)

    with pytest.raises(Exception) as execinfo:
        await remote_action.run(default_dispatcher_collecting, tracker, default_domain)
    assert "you didn't configure an endpoint" in str(execinfo.value)


async def test_remote_action_endpoint_not_running(
    default_dispatcher_collecting, default_domain
):
    tracker = DialogueStateTracker("default", default_domain.slots)

    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with pytest.raises(Exception) as execinfo:
        await remote_action.run(default_dispatcher_collecting, tracker, default_domain)
    assert "Failed to execute custom action." in str(execinfo.value)


async def test_remote_action_endpoint_responds_500(
    default_dispatcher_collecting, default_domain
):
    tracker = DialogueStateTracker("default", default_domain.slots)

    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", status=500)

        with pytest.raises(Exception) as execinfo:
            await remote_action.run(
                default_dispatcher_collecting, tracker, default_domain
            )
        assert "Failed to execute custom action." in str(execinfo.value)


async def test_remote_action_endpoint_responds_400(
    default_dispatcher_collecting, default_domain
):
    tracker = DialogueStateTracker("default", default_domain.slots)

    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with aioresponses() as mocked:
        # noinspection PyTypeChecker
        mocked.post(
            "https://example.com/webhooks/actions",
            exception=ClientResponseError(400, None, '{"action_name": "my_action"}'),
        )

        with pytest.raises(Exception) as execinfo:
            await remote_action.run(
                default_dispatcher_collecting, tracker, default_domain
            )

    assert execinfo.type == ActionExecutionRejection
    assert "Custom action 'my_action' rejected to run" in str(execinfo.value)


async def test_default_action(default_dispatcher_collecting, default_domain):
    tracker = DialogueStateTracker("default", default_domain.slots)

    fallback_action = action.ActionDefaultFallback()

    events = await fallback_action.run(
        default_dispatcher_collecting, tracker, default_domain
    )

    channel = default_dispatcher_collecting.output_channel
    assert channel.messages == [
        {
            "text": "sorry, I didn't get that, can you rephrase it?",
            "recipient_id": "my-sender",
        }
    ]
    assert events == [UserUtteranceReverted()]
