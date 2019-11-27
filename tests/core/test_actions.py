import pytest
from aioresponses import aioresponses

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
    ActionRetrieveResponse,
    RemoteAction,
)
from rasa.core.domain import Domain, InvalidDomain
from rasa.core.events import Restarted, SlotSet, UserUtteranceReverted, BotUttered, Form
from rasa.core.nlg.template import TemplatedNaturalLanguageGenerator
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import ClientResponseError, EndpointConfig
from tests.utilities import json_of_latest_request, latest_request
from rasa.core.constants import UTTER_PREFIX, RESPOND_PREFIX


@pytest.fixture(scope="module")
def template_nlg():
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
def template_sender_tracker(default_domain):
    return DialogueStateTracker("template-sender", default_domain.slots)


def test_text_format():
    assert "{}".format(ActionListen()) == "Action('action_listen')"
    assert (
        "{}".format(ActionUtterTemplate("my_action_name"))
        == "ActionUtterTemplate('my_action_name')"
    )
    assert (
        "{}".format(ActionRetrieveResponse("respond_test"))
        == "ActionRetrieveResponse('respond_test')"
    )


def test_action_instantiation_from_names():
    instantiated_actions = action.actions_from_names(
        ["random_name", "utter_test", "respond_test"],
        None,
        ["random_name", "utter_test"],
    )
    assert len(instantiated_actions) == 3
    assert isinstance(instantiated_actions[0], RemoteAction)
    assert instantiated_actions[0].name() == "random_name"

    assert isinstance(instantiated_actions[1], ActionUtterTemplate)
    assert instantiated_actions[1].name() == "utter_test"

    assert isinstance(instantiated_actions[2], ActionRetrieveResponse)
    assert instantiated_actions[2].name() == "respond_test"


def test_domain_action_instantiation():
    domain = Domain(
        intents={},
        entities=[],
        slots=[],
        templates={},
        action_names=["my_module.ActionTest", "utter_test", "respond_test"],
        form_names=[],
    )

    instantiated_actions = domain.actions(None)

    assert len(instantiated_actions) == 11
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
    assert instantiated_actions[10].name() == "respond_test"


async def test_remote_action_runs(
    default_channel, default_nlg, default_tracker, default_domain
):

    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload={"events": [], "responses": []},
        )

        await remote_action.run(
            default_channel, default_nlg, default_tracker, default_domain
        )

        r = latest_request(mocked, "post", "https://example.com/webhooks/actions")

        assert r

        assert json_of_latest_request(r) == {
            "domain": default_domain.as_dict(),
            "next_action": "my_action",
            "sender_id": "my-sender",
            "version": rasa.__version__,
            "tracker": {
                "latest_message": {
                    "entities": [],
                    "intent": {},
                    "text": None,
                    "message_id": None,
                    "metadata": {},
                },
                "active_form": {},
                "latest_action_name": None,
                "sender_id": "my-sender",
                "paused": False,
                "latest_event_time": None,
                "followup_action": "action_listen",
                "slots": {"name": None},
                "events": [],
                "latest_input_channel": None,
            },
        }


async def test_remote_action_logs_events(
    default_channel, default_nlg, default_tracker, default_domain
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    response = {
        "events": [{"event": "slot", "value": "rasa", "name": "name"}],
        "responses": [
            {
                "text": "test text",
                "template": None,
                "buttons": [{"title": "cheap", "payload": "cheap"}],
            },
            {"template": "utter_greet"},
        ],
    }

    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", payload=response)

        events = await remote_action.run(
            default_channel, default_nlg, default_tracker, default_domain
        )

        r = latest_request(mocked, "post", "https://example.com/webhooks/actions")
        assert r

        assert json_of_latest_request(r) == {
            "domain": default_domain.as_dict(),
            "next_action": "my_action",
            "sender_id": "my-sender",
            "version": rasa.__version__,
            "tracker": {
                "latest_message": {
                    "entities": [],
                    "intent": {},
                    "text": None,
                    "message_id": None,
                    "metadata": {},
                },
                "active_form": {},
                "latest_action_name": None,
                "sender_id": "my-sender",
                "paused": False,
                "followup_action": "action_listen",
                "latest_event_time": None,
                "slots": {"name": None},
                "events": [],
                "latest_input_channel": None,
            },
        }

    assert len(events) == 3  # first two events are bot utterances
    assert events[0] == BotUttered(
        "test text", {"buttons": [{"title": "cheap", "payload": "cheap"}]}
    )
    assert events[1] == BotUttered("hey there None!")
    assert events[2] == SlotSet("name", "rasa")


async def test_remote_action_utterances_with_none_values(
    default_channel, default_tracker, default_domain
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    response = {
        "events": [
            {"event": "form", "name": "restaurant_form", "timestamp": None},
            {
                "event": "slot",
                "timestamp": None,
                "name": "requested_slot",
                "value": "cuisine",
            },
        ],
        "responses": [
            {
                "text": None,
                "buttons": None,
                "elements": [],
                "custom": None,
                "template": "utter_ask_cuisine",
                "image": None,
                "attachment": None,
            }
        ],
    }

    nlg = TemplatedNaturalLanguageGenerator(
        {"utter_ask_cuisine": [{"text": "what dou want to eat?"}]}
    )
    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", payload=response)

        events = await remote_action.run(
            default_channel, nlg, default_tracker, default_domain
        )

    assert events == [
        BotUttered("what dou want to eat?"),
        Form("restaurant_form"),
        SlotSet("requested_slot", "cuisine"),
    ]


async def test_remote_action_without_endpoint(
    default_channel, default_nlg, default_tracker, default_domain
):
    remote_action = action.RemoteAction("my_action", None)

    with pytest.raises(Exception) as execinfo:
        await remote_action.run(
            default_channel, default_nlg, default_tracker, default_domain
        )
    assert "Failed to execute custom action." in str(execinfo.value)


async def test_remote_action_endpoint_not_running(
    default_channel, default_nlg, default_tracker, default_domain
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with pytest.raises(Exception) as execinfo:
        await remote_action.run(
            default_channel, default_nlg, default_tracker, default_domain
        )
    assert "Failed to execute custom action." in str(execinfo.value)


async def test_remote_action_endpoint_responds_500(
    default_channel, default_nlg, default_tracker, default_domain
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", status=500)

        with pytest.raises(Exception) as execinfo:
            await remote_action.run(
                default_channel, default_nlg, default_tracker, default_domain
            )
        assert "Failed to execute custom action." in str(execinfo.value)


async def test_remote_action_endpoint_responds_400(
    default_channel, default_nlg, default_tracker, default_domain
):
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
                default_channel, default_nlg, default_tracker, default_domain
            )

    assert execinfo.type == ActionExecutionRejection
    assert "Custom action 'my_action' rejected to run" in str(execinfo.value)


async def test_action_utter_retrieved_response(
    default_channel, default_nlg, default_tracker, default_domain
):
    from rasa.core.channels.channel import UserMessage

    action_name = "respond_chitchat"
    default_tracker.latest_message = UserMessage(
        "Who are you?",
        parse_data={
            "response_selector": {"chitchat": {"response": {"name": "I am a bot."}}}
        },
    )
    events = await ActionRetrieveResponse(action_name).run(
        default_channel, default_nlg, default_tracker, default_domain
    )

    assert events[0].as_dict().get("text") == BotUttered("I am a bot.").as_dict().get(
        "text"
    )


async def test_action_utter_default_retrieved_response(
    default_channel, default_nlg, default_tracker, default_domain
):
    from rasa.core.channels.channel import UserMessage

    action_name = "respond_chitchat"
    default_tracker.latest_message = UserMessage(
        "Who are you?",
        parse_data={
            "response_selector": {"default": {"response": {"name": "I am a bot."}}}
        },
    )
    events = await ActionRetrieveResponse(action_name).run(
        default_channel, default_nlg, default_tracker, default_domain
    )

    assert events[0].as_dict().get("text") == BotUttered("I am a bot.").as_dict().get(
        "text"
    )


async def test_action_utter_retrieved_empty_response(
    default_channel, default_nlg, default_tracker, default_domain
):
    from rasa.core.channels.channel import UserMessage

    action_name = "respond_chitchat"
    default_tracker.latest_message = UserMessage(
        "Who are you?",
        parse_data={
            "response_selector": {"dummy": {"response": {"name": "I am a bot."}}}
        },
    )
    events = await ActionRetrieveResponse(action_name).run(
        default_channel, default_nlg, default_tracker, default_domain
    )

    assert events == []


async def test_action_utter_template(
    default_channel, default_nlg, default_tracker, default_domain
):
    events = await ActionUtterTemplate("utter_channel").run(
        default_channel, default_nlg, default_tracker, default_domain
    )

    assert events == [BotUttered("this is a default channel")]


async def test_action_utter_template_unknown_template(
    default_channel, default_nlg, default_tracker, default_domain
):
    events = await ActionUtterTemplate("utter_unknown").run(
        default_channel, default_nlg, default_tracker, default_domain
    )

    assert events == []


async def test_action_utter_template_with_buttons(
    default_channel, template_nlg, template_sender_tracker, default_domain
):
    events = await ActionUtterTemplate("utter_buttons").run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )

    assert events == [
        BotUttered(
            "button message",
            {
                "buttons": [
                    {"payload": "button1", "title": "button1"},
                    {"payload": "button2", "title": "button2"},
                ]
            },
        )
    ]


async def test_action_utter_template_invalid_template(
    default_channel, template_nlg, template_sender_tracker, default_domain
):
    events = await ActionUtterTemplate("utter_invalid").run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )

    assert len(events) == 1
    assert isinstance(events[0], BotUttered)
    assert events[0].text.startswith("a template referencing an invalid {variable}.")


async def test_action_utter_template_channel_specific(
    default_nlg, default_tracker, default_domain
):
    from rasa.core.channels.slack import SlackBot

    output_channel = SlackBot("DummyToken", "General")

    events = await ActionUtterTemplate("utter_channel").run(
        output_channel, default_nlg, default_tracker, default_domain
    )

    assert events == [
        BotUttered("you're talking to me on slack!", metadata={"channel": "slack"})
    ]


async def test_action_back(
    default_channel, template_nlg, template_sender_tracker, default_domain
):
    events = await ActionBack().run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )

    assert events == [
        BotUttered("backing up..."),
        UserUtteranceReverted(),
        UserUtteranceReverted(),
    ]


async def test_action_restart(
    default_channel, template_nlg, template_sender_tracker, default_domain
):
    events = await ActionRestart().run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )

    assert events == [BotUttered("congrats, you've restarted me!"), Restarted()]


async def test_action_default_fallback(
    default_channel, default_nlg, default_tracker, default_domain
):
    events = await ActionDefaultFallback().run(
        default_channel, default_nlg, default_tracker, default_domain
    )

    assert events == [
        BotUttered("sorry, I didn't get that, can you rephrase it?"),
        UserUtteranceReverted(),
    ]


async def test_action_default_ask_affirmation(
    default_channel, default_nlg, default_tracker, default_domain
):
    events = await ActionDefaultAskAffirmation().run(
        default_channel, default_nlg, default_tracker, default_domain
    )

    assert events == [
        BotUttered(
            "Did you mean 'None'?",
            {
                "buttons": [
                    {"title": "Yes", "payload": "/None"},
                    {"title": "No", "payload": "/out_of_scope"},
                ]
            },
        )
    ]


async def test_action_default_ask_rephrase(
    default_channel, template_nlg, template_sender_tracker, default_domain
):
    events = await ActionDefaultAskRephrase().run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )

    assert events == [BotUttered("can you rephrase that?")]
