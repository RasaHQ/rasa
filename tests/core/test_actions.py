import pytest
from aioresponses import aioresponses
from httpretty import httpretty

import rasa.core
from rasa.core.actions import action
from rasa.core.actions.action import (
    ACTION_BACK_NAME,
    ACTION_DEACTIVATE_FORM_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ActionBack,
    ActionDefaultAskAffirmation,
    ActionDefaultAskRephrase,
    ActionDefaultFallback,
    ActionExecutionRejection,
    ActionListen,
    ActionRestart,
    ActionUtterTemplate,
    RemoteAction,
    send_response,
)
from rasa.core.channels.channel import CollectingOutputChannel
from rasa.core.domain import Domain
from rasa.core.events import Restarted, SlotSet, UserUtteranceReverted
from rasa.core.nlg.template import TemplatedNaturalLanguageGenerator
from rasa.core.trackers import DialogueStateTracker
from rasa.core.processor import Dispatcher
from rasa.utils.endpoints import ClientResponseError, EndpointConfig
from tests.utilities import json_of_latest_request, latest_request


@pytest.fixture(scope="module")
def default_template_nlg():
    templates = {
        "utter_ask_rephrase": [{"text": "can you rephrase that?"}],
        "utter_restart": [{"text": "congrats, you've restarted me!"}],
        "utter_back": [{"text": "backing up..."}],
        "utter_invalid": [{"text": "a template referencing an invalid {variable}."}],
        "utter_buttons": [
            {
                "text": "button message",
                "buttons": [
                    {"payload": "button1", "title": "button1"},
                    {"payload": "button2", "title": "button2"},
                ],
            }
        ],
    }
    return TemplatedNaturalLanguageGenerator(templates)


@pytest.fixture(scope="module")
def default_template_dispatcher():
    bot = CollectingOutputChannel()
    return Dispatcher("template-sender", bot, default_template_nlg())


def test_text_format():
    assert "{}".format(ActionListen()) == "Action('action_listen')"
    assert (
        "{}".format(ActionUtterTemplate("my_action_name"))
        == "ActionUtterTemplate('my_action_name')"
    )


def test_action_instantiation_from_names():
    instantiated_actions = action.actions_from_names(
        ["random_name", "utter_test"], None, ["random_name", "utter_test"]
    )
    assert len(instantiated_actions) == 2
    assert isinstance(instantiated_actions[0], RemoteAction)
    assert instantiated_actions[0].name() == "random_name"

    assert isinstance(instantiated_actions[1], ActionUtterTemplate)
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


async def test_send_response(default_dispatcher_collecting):
    text_only_message = {"text": "hey"}
    image_only_message = {"image": "https://i.imgur.com/nGF1K8f.jpg"}
    text_and_image_message = {
        "text": "look at this",
        "image": "https://i.imgur.com/T5xVo.jpg",
    }

    await send_response(default_dispatcher_collecting, text_only_message)
    await send_response(default_dispatcher_collecting, image_only_message)
    await send_response(default_dispatcher_collecting, text_and_image_message)
    collected = default_dispatcher_collecting.output_channel.messages

    assert len(collected) == 4

    # text only message
    assert collected[0] == {"recipient_id": "my-sender", "text": "hey"}

    # image only message
    assert collected[1] == {
        "recipient_id": "my-sender",
        "image": "https://i.imgur.com/nGF1K8f.jpg",
    }

    # text & image combined - will result in two messages
    assert collected[2] == {"recipient_id": "my-sender", "text": "look at this"}
    assert collected[3] == {
        "recipient_id": "my-sender",
        "image": "https://i.imgur.com/T5xVo.jpg",
    }


async def test_action_utter_template(
    default_dispatcher_collecting, default_tracker, default_domain
):
    dispatcher = default_dispatcher_collecting

    events = await ActionUtterTemplate("utter_channel").run(
        dispatcher, default_tracker, default_domain
    )

    assert dispatcher.output_channel.latest_output() == {
        "text": "this is a default channel",
        "recipient_id": "my-sender",
    }
    assert events == []


async def test_action_utter_template_unknown_template(
    default_dispatcher_collecting, default_tracker, default_domain
):
    dispatcher = default_dispatcher_collecting
    # TODO add real intent

    events = await ActionUtterTemplate("utter_unknown").run(
        dispatcher, default_tracker, default_domain
    )

    assert dispatcher.output_channel.latest_output() is None
    assert events == []


async def test_action_utter_template_with_buttons(
    default_template_dispatcher, default_tracker, default_domain
):
    dispatcher = default_template_dispatcher

    events = await ActionUtterTemplate("utter_buttons").run(
        dispatcher, default_tracker, default_domain
    )

    assert dispatcher.output_channel.latest_output() == {
        "text": "button message",
        "buttons": [
            {"payload": "button1", "title": "button1"},
            {"payload": "button2", "title": "button2"},
        ],
        "recipient_id": "template-sender",
    }
    assert events == []


async def test_action_utter_template_invalid_template(
    default_template_dispatcher, default_tracker, default_domain
):
    dispatcher = default_template_dispatcher

    events = await ActionUtterTemplate("utter_invalid").run(
        dispatcher, default_tracker, default_domain
    )

    collected = dispatcher.output_channel.latest_output()
    assert collected["text"].startswith("a template referencing an invalid {variable}.")
    assert events == []


async def test_action_utter_template_channel_specific(
    default_nlg, default_tracker, default_domain
):
    from rasa.core.channels.slack import SlackBot

    httpretty.register_uri(
        httpretty.POST,
        "https://slack.com/api/chat.postMessage",
        body='{"ok":true,"purpose":"Testing bots"}',
    )
    httpretty.enable()

    bot = SlackBot("DummyToken", "General")
    dispatcher = Dispatcher("my-sender", bot, default_nlg)

    events = await ActionUtterTemplate("utter_channel").run(
        dispatcher, default_tracker, default_domain
    )
    httpretty.disable()

    r = httpretty.latest_requests[-1]
    assert r.parsed_body == {
        "as_user": ["True"],
        "channel": ["General"],
        "text": ["you're talking to me on slack!"],
    }
    assert events == []


async def test_action_back(
    default_template_dispatcher, default_tracker, default_domain
):
    dispatcher = default_template_dispatcher

    events = await ActionBack().run(dispatcher, default_tracker, default_domain)

    assert dispatcher.output_channel.latest_output() == {
        "text": "backing up...",
        "recipient_id": "template-sender",
    }
    assert events == [UserUtteranceReverted(), UserUtteranceReverted()]


async def test_action_restart(
    default_template_dispatcher, default_tracker, default_domain
):
    dispatcher = default_template_dispatcher

    events = await ActionRestart().run(dispatcher, default_tracker, default_domain)

    assert dispatcher.output_channel.latest_output() == {
        "text": "congrats, you've restarted me!",
        "recipient_id": "template-sender",
    }

    assert events == [Restarted()]


async def test_action_default_fallback(
    default_dispatcher_collecting, default_tracker, default_domain
):
    dispatcher = default_dispatcher_collecting

    events = await ActionDefaultFallback().run(
        dispatcher, default_tracker, default_domain
    )

    assert dispatcher.output_channel.latest_output() == {
        "text": "sorry, I didn't get that, can you rephrase it?",
        "recipient_id": "my-sender",
    }
    assert events == [UserUtteranceReverted()]


async def test_action_default_ask_affirmation(
    default_dispatcher_collecting, default_tracker, default_domain
):
    dispatcher = default_dispatcher_collecting
    # TODO add real intent

    events = await ActionDefaultAskAffirmation().run(
        dispatcher, default_tracker, default_domain
    )

    assert dispatcher.output_channel.messages == [
        {
            "text": "Did you mean 'None'?",
            "buttons": [
                {"title": "Yes", "payload": "/None"},
                {"title": "No", "payload": "/out_of_scope"},
            ],
            "recipient_id": "my-sender",
        }
    ]
    assert events == []


async def test_action_default_ask_rephrase(
    default_template_dispatcher, default_tracker, default_domain
):
    dispatcher = default_template_dispatcher

    events = await ActionDefaultAskRephrase().run(
        dispatcher, default_tracker, default_domain
    )

    assert dispatcher.output_channel.latest_output() == {
        "text": "can you rephrase that?",
        "recipient_id": "template-sender",
    }
    assert events == []
