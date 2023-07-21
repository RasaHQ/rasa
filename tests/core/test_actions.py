import logging
import textwrap
from datetime import datetime
from typing import List, Text, Any, Dict, Optional
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch
from _pytest.logging import LogCaptureFixture
from aioresponses import aioresponses
from jsonschema import ValidationError

import rasa.core
from rasa.core.actions import action
from rasa.core.actions.action import (
    ActionBack,
    ActionDefaultAskAffirmation,
    ActionDefaultAskRephrase,
    ActionDefaultFallback,
    ActionExecutionRejection,
    ActionRestart,
    ActionBotResponse,
    ActionRetrieveResponse,
    RemoteAction,
    ActionSessionStart,
    ActionEndToEndResponse,
    ActionExtractSlots,
)
from rasa.core.actions.forms import FormAction
from rasa.core.channels import CollectingOutputChannel, OutputChannel
from rasa.core.channels.slack import SlackBot
from rasa.core.constants import COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.constants import (
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    UTTER_PREFIX,
    REQUIRED_SLOTS_KEY,
)
from rasa.shared.core.domain import (
    ActionNotFoundException,
    SessionConfig,
    Domain,
    KEY_E2E_ACTIONS,
)
from rasa.shared.core.events import (
    Restarted,
    SlotSet,
    UserUtteranceReverted,
    BotUttered,
    ActiveLoop,
    SessionStarted,
    ActionExecuted,
    Event,
    UserUttered,
    EntitiesAdded,
    DefinePrevUserUtteredFeaturization,
    AllSlotsReset,
    ReminderScheduled,
    ReminderCancelled,
    ActionReverted,
    StoryExported,
    FollowupAction,
    ConversationPaused,
    ConversationResumed,
    AgentUttered,
    LoopInterrupted,
    ActionExecutionRejected,
    LegacyFormValidation,
    LegacyForm,
)
import rasa.shared.utils.common
from rasa.core.nlg.response import TemplatedNaturalLanguageGenerator
from rasa.shared.core.constants import (
    USER_INTENT_SESSION_START,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_DEACTIVATE_LOOP_NAME,
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_BACK_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
    RULE_SNIPPET_ACTION_NAME,
    ACTIVE_LOOP,
    FOLLOWUP_ACTION,
    REQUESTED_SLOT,
    SESSION_START_METADATA_SLOT,
    ACTION_EXTRACT_SLOTS,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import ClientResponseError, EndpointConfig
from tests.utilities import json_of_latest_request, latest_request


@pytest.fixture(scope="module")
def template_nlg() -> TemplatedNaturalLanguageGenerator:
    responses = {
        "utter_ask_rephrase": [{"text": "can you rephrase that?"}],
        "utter_restart": [{"text": "congrats, you've restarted me!"}],
        "utter_back": [{"text": "backing up..."}],
        "utter_invalid": [{"text": "a response referencing an invalid {variable}."}],
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
    return TemplatedNaturalLanguageGenerator(responses)


@pytest.fixture(scope="module")
def template_sender_tracker(domain_path: Text):
    domain = Domain.load(domain_path)
    return DialogueStateTracker("template-sender", domain.slots)


def test_domain_action_instantiation():
    domain = Domain(
        intents=[{"chitchat": {"is_retrieval_intent": True}}],
        entities=[],
        slots=[],
        responses={},
        action_names=["my_module.ActionTest", "utter_test", "utter_chitchat"],
        forms={},
        data={},
    )

    instantiated_actions = [
        action.action_for_name_or_text(action_name, domain, None)
        for action_name in domain.action_names_or_texts
    ]

    assert len(instantiated_actions) == 16
    assert instantiated_actions[0].name() == ACTION_LISTEN_NAME
    assert instantiated_actions[1].name() == ACTION_RESTART_NAME
    assert instantiated_actions[2].name() == ACTION_SESSION_START_NAME
    assert instantiated_actions[3].name() == ACTION_DEFAULT_FALLBACK_NAME
    assert instantiated_actions[4].name() == ACTION_DEACTIVATE_LOOP_NAME
    assert instantiated_actions[5].name() == ACTION_REVERT_FALLBACK_EVENTS_NAME
    assert instantiated_actions[6].name() == ACTION_DEFAULT_ASK_AFFIRMATION_NAME
    assert instantiated_actions[7].name() == ACTION_DEFAULT_ASK_REPHRASE_NAME
    assert instantiated_actions[8].name() == ACTION_TWO_STAGE_FALLBACK_NAME
    assert instantiated_actions[9].name() == ACTION_UNLIKELY_INTENT_NAME
    assert instantiated_actions[10].name() == ACTION_BACK_NAME
    assert instantiated_actions[11].name() == RULE_SNIPPET_ACTION_NAME
    assert instantiated_actions[12].name() == ACTION_EXTRACT_SLOTS
    assert instantiated_actions[13].name() == "my_module.ActionTest"
    assert instantiated_actions[14].name() == "utter_test"
    assert instantiated_actions[15].name() == "utter_chitchat"


@pytest.mark.parametrize(
    "is_compression_enabled, expected_compress_argument",
    [
        ("True", True),
        ("False", False),
    ],
)
async def test_remote_actions_are_compressed(
    is_compression_enabled: str,
    expected_compress_argument: bool,
    default_channel: OutputChannel,
    default_nlg: NaturalLanguageGenerator,
    default_tracker: DialogueStateTracker,
    domain: Domain,
    monkeypatch: MonkeyPatch,
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)
    monkeypatch.setenv(COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME, is_compression_enabled)

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload={"events": [], "responses": []},
        )

        await remote_action.run(default_channel, default_nlg, default_tracker, domain)

        r = latest_request(mocked, "post", "https://example.com/webhooks/actions")

        assert r
        assert r[-1].kwargs["compress"] is expected_compress_argument


async def test_remote_action_runs(
    default_channel: OutputChannel,
    default_nlg: NaturalLanguageGenerator,
    default_tracker: DialogueStateTracker,
    domain: Domain,
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload={"events": [], "responses": []},
        )

        await remote_action.run(default_channel, default_nlg, default_tracker, domain)

        r = latest_request(mocked, "post", "https://example.com/webhooks/actions")

        assert r

        assert json_of_latest_request(r) == {
            "domain": domain.as_dict(),
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
                ACTIVE_LOOP: {},
                "latest_action": {},
                "latest_action_name": None,
                "sender_id": "my-sender",
                "paused": False,
                "latest_event_time": None,
                FOLLOWUP_ACTION: "action_listen",
                "slots": {
                    "name": None,
                    REQUESTED_SLOT: None,
                    SESSION_START_METADATA_SLOT: None,
                },
                "events": [],
                "latest_input_channel": None,
            },
        }


async def test_remote_action_logs_events(
    default_channel: OutputChannel,
    default_nlg: NaturalLanguageGenerator,
    default_tracker: DialogueStateTracker,
    domain: Domain,
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    response = {
        "events": [{"event": "slot", "value": "rasa", "name": "name"}],
        "responses": [
            {
                "text": "test text",
                "response": None,
                "buttons": [{"title": "cheap", "payload": "cheap"}],
            },
            {"response": "utter_greet"},
        ],
    }

    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", payload=response)

        events = await remote_action.run(
            default_channel, default_nlg, default_tracker, domain
        )

        r = latest_request(mocked, "post", "https://example.com/webhooks/actions")
        assert r

        assert json_of_latest_request(r) == {
            "domain": domain.as_dict(),
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
                ACTIVE_LOOP: {},
                "latest_action": {},
                "latest_action_name": None,
                "sender_id": "my-sender",
                "paused": False,
                FOLLOWUP_ACTION: ACTION_LISTEN_NAME,
                "latest_event_time": None,
                "slots": {
                    "name": None,
                    REQUESTED_SLOT: None,
                    SESSION_START_METADATA_SLOT: None,
                },
                "events": [],
                "latest_input_channel": None,
            },
        }
    assert len(events) == 3  # first two events are bot utterances
    assert events[0] == BotUttered(
        "test text", {"buttons": [{"title": "cheap", "payload": "cheap"}]}
    )
    assert events[1] == BotUttered(
        "hey there None!", metadata={"utter_action": "utter_greet"}
    )
    assert events[2] == SlotSet("name", "rasa")


async def test_remote_action_utterances_with_none_values(
    default_channel: OutputChannel,
    default_tracker: DialogueStateTracker,
    domain: Domain,
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
                "response": "utter_ask_cuisine",
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

        events = await remote_action.run(default_channel, nlg, default_tracker, domain)

    assert events == [
        BotUttered(
            "what dou want to eat?", metadata={"utter_action": "utter_ask_cuisine"}
        ),
        ActiveLoop("restaurant_form"),
        SlotSet("requested_slot", "cuisine"),
    ]


@pytest.mark.parametrize(
    "event",
    (
        EntitiesAdded(
            entities=[
                {"entity": "city", "value": "London"},
                {"entity": "count", "value": 1},
            ],
            timestamp=None,
        ),
        EntitiesAdded(entities=[]),
        EntitiesAdded(
            entities=[
                {"entity": "name", "value": "John", "role": "contact", "group": "test"}
            ]
        ),
        DefinePrevUserUtteredFeaturization(
            use_text_for_featurization=False, timestamp=None, metadata=None
        ),
        ReminderCancelled(timestamp=1621590172.3872123),
        ReminderScheduled(
            timestamp=None, trigger_date_time=datetime.now(), intent="greet"
        ),
        ActionExecutionRejected(action_name="my_action"),
        LegacyFormValidation(validate=True, timestamp=None),
        LoopInterrupted(timestamp=None, is_interrupted=False),
        ActiveLoop(name="loop"),
        LegacyForm(name="my_form"),
        AllSlotsReset(),
        SlotSet(key="my_slot", value={}),
        SlotSet(key="my slot", value=[]),
        SlotSet(key="test", value=1),
        SlotSet(key="test", value="text"),
        ConversationResumed(),
        ConversationPaused(),
        FollowupAction(name="test"),
        StoryExported(),
        Restarted(),
        ActionReverted(),
        UserUtteranceReverted(),
        BotUttered(text="Test bot utterance"),
        UserUttered(
            parse_data={
                "entities": [],
                "response_selector": {
                    "all_retrieval_intents": [],
                    "chitchat/ask_weather": {"response": {}, "ranking": []},
                },
            }
        ),
        UserUttered(
            text="hello",
            parse_data={
                "intent": {"name": "greet", "confidence": 0.9604260921478271},
                "entities": [
                    {"entity": "city", "value": "London"},
                    {"entity": "count", "value": 1},
                ],
                "text": "hi",
                "message_id": "3f4c04602a4947098c574b107d3ccc50",
                "metadata": {},
                "intent_ranking": [
                    {"name": "greet", "confidence": 0.9604260921478271},
                    {"name": "goodbye", "confidence": 0.01835782080888748},
                    {"name": "deny", "confidence": 0.011255578137934208},
                    {"name": "bot_challenge", "confidence": 0.004019865766167641},
                    {"name": "affirm", "confidence": 0.002524246694520116},
                    {"name": "mood_great", "confidence": 0.002214624546468258},
                    {"name": "chitchat", "confidence": 0.0009614597074687481},
                    {"name": "mood_unhappy", "confidence": 0.00024030178610701114},
                ],
                "response_selector": {
                    "all_retrieval_intents": [],
                    "default": {
                        "response": {
                            "id": -226546773594344189,
                            "responses": [{"text": "chitchat/ask_name"}],
                            "response_templates": [{"text": "chitchat/ask_name"}],
                            "confidence": 0.9618658423423767,
                            "intent_response_key": "chitchat/ask_name",
                            "utter_action": "utter_chitchat/ask_name",
                            "template_name": "utter_chitchat/ask_name",
                        },
                        "ranking": [
                            {
                                "id": -226546773594344189,
                                "confidence": 0.9618658423423767,
                                "intent_response_key": "chitchat/ask_name",
                            },
                            {
                                "id": 8392727822750416828,
                                "confidence": 0.03813415765762329,
                                "intent_response_key": "chitchat/ask_weather",
                            },
                        ],
                    },
                },
            },
        ),
        SessionStarted(),
        ActionExecuted(action_name="action_listen"),
        AgentUttered(),
    ),
)
async def test_remote_action_valid_payload_all_events(
    default_channel: OutputChannel,
    default_nlg: NaturalLanguageGenerator,
    default_tracker: DialogueStateTracker,
    domain: Domain,
    event: Event,
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)
    events = [event.as_dict()]
    response = {"events": events, "responses": []}
    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", payload=response)

        events = await remote_action.run(
            default_channel, default_nlg, default_tracker, domain
        )

    assert len(events) == 1


@pytest.mark.parametrize(
    "event",
    (
        {
            "event": "user",
            "timestamp": 1621590172.3872123,
            "parse_data": {"entities": {}},
        },
        {"event": "entities", "timestamp": 1621604905.647361, "entities": {}},
    ),
)
async def test_remote_action_invalid_entities_payload(
    default_channel: OutputChannel,
    default_nlg: NaturalLanguageGenerator,
    default_tracker: DialogueStateTracker,
    domain: Domain,
    event: Event,
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)
    response = {"events": [event], "responses": []}
    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", payload=response)

        with pytest.raises(ValidationError) as e:
            await remote_action.run(
                default_channel, default_nlg, default_tracker, domain
            )

    assert "Failed to validate Action server response from API" in str(e.value)


async def test_remote_action_multiple_events_payload(
    default_channel: OutputChannel,
    default_nlg: NaturalLanguageGenerator,
    default_tracker: DialogueStateTracker,
    domain: Domain,
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)
    response = {
        "events": [
            {
                "event": "action",
                "name": "action_listen",
                "policy": None,
                "confidence": None,
                "timestamp": None,
            },
            {"event": "slot", "name": "name", "value": None, "timestamp": None},
            {
                "event": "user",
                "timestamp": None,
                "text": "hello",
                "parse_data": {
                    "intent": {"name": "greet", "confidence": 0.99},
                    "entities": [],
                },
            },
        ],
        "responses": [],
    }

    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", payload=response)

        events = await remote_action.run(
            default_channel, default_nlg, default_tracker, domain
        )

    assert isinstance(events[0], ActionExecuted)
    assert events[0].as_dict().get("name") == "action_listen"

    assert isinstance(events[1], SlotSet)
    assert events[1].as_dict().get("name") == "name"

    assert isinstance(events[2], UserUttered)
    assert events[2].as_dict().get("text") == "hello"


async def test_remote_action_without_endpoint(
    default_channel, default_nlg, default_tracker, domain: Domain
):
    remote_action = action.RemoteAction("my_action", None)

    with pytest.raises(Exception) as execinfo:
        await remote_action.run(default_channel, default_nlg, default_tracker, domain)
    assert "Failed to execute custom action" in str(execinfo.value)


async def test_remote_action_endpoint_not_running(
    default_channel, default_nlg, default_tracker, domain: Domain
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with pytest.raises(Exception) as execinfo:
        await remote_action.run(default_channel, default_nlg, default_tracker, domain)
    assert "Failed to execute custom action" in str(execinfo.value)


async def test_remote_action_endpoint_responds_500(
    default_channel, default_nlg, default_tracker, domain: Domain
):
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    remote_action = action.RemoteAction("my_action", endpoint)

    with aioresponses() as mocked:
        mocked.post("https://example.com/webhooks/actions", status=500)

        with pytest.raises(Exception) as execinfo:
            await remote_action.run(
                default_channel, default_nlg, default_tracker, domain
            )
        assert "Failed to execute custom action" in str(execinfo.value)


async def test_remote_action_endpoint_responds_400(
    default_channel, default_nlg, default_tracker, domain: Domain
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
                default_channel, default_nlg, default_tracker, domain
            )

    assert execinfo.type == ActionExecutionRejection
    assert "Custom action 'my_action' rejected to run" in str(execinfo.value)


async def test_action_utter_retrieved_response(
    default_channel, default_nlg, default_tracker, domain: Domain
):
    from rasa.core.channels.channel import UserMessage

    action_name = "utter_chitchat"
    default_tracker.latest_message = UserMessage(
        "Who are you?",
        parse_data={
            "response_selector": {
                "chitchat": {
                    "response": {
                        "intent_response_key": "chitchat/ask_name",
                        "responses": [{"text": "I am a bot."}],
                        "utter_action": "utter_chitchat/ask_name",
                    }
                }
            }
        },
    )

    domain.responses.update({"utter_chitchat/ask_name": [{"text": "I am a bot."}]})

    events = await ActionRetrieveResponse(action_name).run(
        default_channel, default_nlg, default_tracker, domain
    )

    assert events[0].as_dict().get("text") == BotUttered("I am a bot.").as_dict().get(
        "text"
    )
    assert (
        events[0].as_dict().get("metadata").get("utter_action")
        == "utter_chitchat/ask_name"
    )


async def test_action_utter_default_retrieved_response(
    default_channel, default_nlg, default_tracker, domain: Domain
):
    from rasa.core.channels.channel import UserMessage

    action_name = "utter_chitchat"
    default_tracker.latest_message = UserMessage(
        "Who are you?",
        parse_data={
            "response_selector": {
                "default": {
                    "response": {
                        "intent_response_key": "chitchat/ask_name",
                        "responses": [{"text": "I am a bot."}],
                        "utter_action": "utter_chitchat/ask_name",
                    }
                }
            }
        },
    )

    domain.responses.update({"utter_chitchat/ask_name": [{"text": "I am a bot."}]})

    events = await ActionRetrieveResponse(action_name).run(
        default_channel, default_nlg, default_tracker, domain
    )

    assert events[0].as_dict().get("text") == BotUttered("I am a bot.").as_dict().get(
        "text"
    )

    assert (
        events[0].as_dict().get("metadata").get("utter_action")
        == "utter_chitchat/ask_name"
    )


async def test_action_utter_retrieved_empty_response(
    default_channel, default_nlg, default_tracker, domain: Domain
):
    from rasa.core.channels.channel import UserMessage

    action_name = "utter_chitchat"
    default_tracker.latest_message = UserMessage(
        "Who are you?",
        parse_data={
            "response_selector": {
                "dummy": {
                    "response": {
                        "intent_response_key": "chitchat/ask_name",
                        "responses": [{"text": "I am a bot."}],
                        "utter_action": "utter_chitchat/ask_name",
                    }
                }
            }
        },
    )

    domain.responses.update({"utter_chitchat/ask_name": [{"text": "I am a bot."}]})

    events = await ActionRetrieveResponse(action_name).run(
        default_channel, default_nlg, default_tracker, domain
    )

    assert events == []


async def test_response(default_channel, default_nlg, default_tracker, domain: Domain):
    events = await ActionBotResponse("utter_channel").run(
        default_channel, default_nlg, default_tracker, domain
    )

    assert events == [
        BotUttered(
            "this is a default channel", metadata={"utter_action": "utter_channel"}
        )
    ]


async def test_response_unknown_response(
    default_channel, default_nlg, default_tracker, domain: Domain
):
    events = await ActionBotResponse("utter_unknown").run(
        default_channel, default_nlg, default_tracker, domain
    )

    assert events == []


async def test_response_with_buttons(
    default_channel, template_nlg, template_sender_tracker, domain: Domain
):
    events = await ActionBotResponse("utter_buttons").run(
        default_channel, template_nlg, template_sender_tracker, domain
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
            metadata={"utter_action": "utter_buttons"},
        )
    ]


async def test_response_invalid_response(
    default_channel, template_nlg, template_sender_tracker, domain: Domain
):
    events = await ActionBotResponse("utter_invalid").run(
        default_channel, template_nlg, template_sender_tracker, domain
    )

    assert len(events) == 1
    assert isinstance(events[0], BotUttered)
    assert events[0].text.startswith("a response referencing an invalid {variable}.")


async def test_response_channel_specific(default_nlg, default_tracker, domain: Domain):

    output_channel = SlackBot("DummyToken", "General")

    events = await ActionBotResponse("utter_channel").run(
        output_channel, default_nlg, default_tracker, domain
    )

    assert events == [
        BotUttered(
            "you're talking to me on slack!",
            metadata={"channel": "slack", "utter_action": "utter_channel"},
        )
    ]


@pytest.fixture
def domain_with_response_ids() -> Domain:
    domain_yaml = """
    responses:
        utter_one_id:
            - text: test
              id: '1'
        utter_multiple_ids:
            - text: test
              id: '2'
            - text: test
              id: '3'
        utter_no_id:
            - text: test
    """
    domain = Domain.from_yaml(domain_yaml)
    return domain


async def test_response_with_response_id(
    default_channel, domain_with_response_ids: Domain
) -> None:
    nlg = TemplatedNaturalLanguageGenerator(domain_with_response_ids.responses)

    events = await ActionBotResponse("utter_one_id").run(
        default_channel,
        nlg,
        DialogueStateTracker("response_id", slots=[]),
        domain_with_response_ids,
    )

    assert events == [
        BotUttered(
            "test",
            metadata={"id": "1", "utter_action": "utter_one_id"},
        )
    ]


async def test_action_back(
    default_channel, template_nlg, template_sender_tracker, domain: Domain
):
    events = await ActionBack().run(
        default_channel, template_nlg, template_sender_tracker, domain
    )

    assert events == [
        BotUttered("backing up...", metadata={"utter_action": "utter_back"}),
        UserUtteranceReverted(),
        UserUtteranceReverted(),
    ]


async def test_action_restart(
    default_channel, template_nlg, template_sender_tracker, domain: Domain
):
    events = await ActionRestart().run(
        default_channel, template_nlg, template_sender_tracker, domain
    )

    assert events == [
        BotUttered(
            "congrats, you've restarted me!", metadata={"utter_action": "utter_restart"}
        ),
        Restarted(),
    ]


async def test_action_session_start_without_slots(
    default_channel: CollectingOutputChannel,
    template_nlg: TemplatedNaturalLanguageGenerator,
    template_sender_tracker: DialogueStateTracker,
    domain: Domain,
):
    events = await ActionSessionStart().run(
        default_channel, template_nlg, template_sender_tracker, domain
    )
    assert events == [SessionStarted(), ActionExecuted(ACTION_LISTEN_NAME)]


@pytest.mark.parametrize(
    "session_config, expected_events",
    [
        (
            SessionConfig(123, True),
            [
                SessionStarted(),
                SlotSet("my_slot", "value"),
                SlotSet("another-slot", "value2"),
                ActionExecuted(action_name=ACTION_LISTEN_NAME),
            ],
        ),
        (
            SessionConfig(123, False),
            [SessionStarted(), ActionExecuted(action_name=ACTION_LISTEN_NAME)],
        ),
    ],
)
async def test_action_session_start_with_slots(
    default_channel: CollectingOutputChannel,
    template_nlg: TemplatedNaturalLanguageGenerator,
    template_sender_tracker: DialogueStateTracker,
    domain: Domain,
    session_config: SessionConfig,
    expected_events: List[Event],
):
    # set a few slots on tracker
    slot_set_event_1 = SlotSet("my_slot", "value")
    slot_set_event_2 = SlotSet("another-slot", "value2")
    for event in [slot_set_event_1, slot_set_event_2]:
        template_sender_tracker.update(event)

    domain.session_config = session_config

    events = await ActionSessionStart().run(
        default_channel, template_nlg, template_sender_tracker, domain
    )

    assert events == expected_events

    # make sure that the list of events has ascending timestamps
    assert sorted(events, key=lambda x: x.timestamp) == events


async def test_applied_events_after_action_session_start(
    default_channel: CollectingOutputChannel,
    template_nlg: TemplatedNaturalLanguageGenerator,
):
    slot_set = SlotSet("my_slot", "value")
    events = [
        slot_set,
        ActionExecuted(ACTION_LISTEN_NAME),
        # User triggers a restart manually by triggering the intent
        UserUttered(
            text=f"/{USER_INTENT_SESSION_START}",
            intent={"name": USER_INTENT_SESSION_START},
        ),
    ]
    tracker = DialogueStateTracker.from_events("🕵️‍♀️", events)

    # Mapping Policy kicks in and runs the session restart action
    events = await ActionSessionStart().run(
        default_channel, template_nlg, tracker, Domain.empty()
    )
    for event in events:
        tracker.update(event)

    assert tracker.applied_events() == [slot_set, ActionExecuted(ACTION_LISTEN_NAME)]


async def test_action_default_fallback(
    default_channel, default_nlg, default_tracker, domain: Domain
):
    events = await ActionDefaultFallback().run(
        default_channel, default_nlg, default_tracker, domain
    )

    assert events == [
        BotUttered(
            "sorry, I didn't get that, can you rephrase it?",
            metadata={"utter_action": "utter_default"},
        ),
        UserUtteranceReverted(),
    ]


async def test_action_default_ask_affirmation(
    default_channel, default_nlg, domain: Domain
):
    initial_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        # User triggers a restart manually by triggering the intent
        UserUttered(
            text="/foobar",
            intent={"name": "foobar"},
            parse_data={
                "intent_ranking": [
                    {"confidence": 0.9, "name": "foobar"},
                    {"confidence": 0.1, "name": "baz"},
                ]
            },
        ),
    ]
    tracker = DialogueStateTracker.from_events("🕵️‍♀️", initial_events)

    events = await ActionDefaultAskAffirmation().run(
        default_channel, default_nlg, tracker, domain
    )

    assert events == [
        BotUttered(
            "Did you mean 'foobar'?",
            {
                "buttons": [
                    {"title": "Yes", "payload": "/foobar"},
                    {"title": "No", "payload": "/out_of_scope"},
                ]
            },
            {"utter_action": "action_default_ask_affirmation"},
        )
    ]


async def test_action_default_ask_affirmation_on_empty_conversation(
    default_channel, default_nlg, default_tracker, domain: Domain
):
    events = await ActionDefaultAskAffirmation().run(
        default_channel, default_nlg, default_tracker, domain
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
            {"utter_action": "action_default_ask_affirmation"},
        )
    ]


async def test_action_default_ask_rephrase(
    default_channel, template_nlg, template_sender_tracker, domain: Domain
):
    events = await ActionDefaultAskRephrase().run(
        default_channel, template_nlg, template_sender_tracker, domain
    )

    assert events == [
        BotUttered(
            "can you rephrase that?", metadata={"utter_action": "utter_ask_rephrase"}
        )
    ]


@pytest.mark.parametrize(
    "slot",
    [
        """- my_slot
        """,
        "[]",
    ],
)
def test_get_form_action(slot: Text):
    form_action_name = "my_business_logic"

    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
    slots:
      my_slot:
        type: text
        mappings:
        - type: from_text
    actions:
    - my_action
    forms:
      {form_action_name}:
        {REQUIRED_SLOTS_KEY}:
          {slot}
    """
        )
    )

    actual = action.action_for_name_or_text(form_action_name, domain, None)
    assert isinstance(actual, FormAction)


def test_overridden_form_action():
    form_action_name = "my_business_logic"
    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
    actions:
    - my_action
    - {form_action_name}
    forms:
        {form_action_name}:
          {REQUIRED_SLOTS_KEY}: []
    """
        )
    )

    actual = action.action_for_name_or_text(form_action_name, domain, None)
    assert isinstance(actual, RemoteAction)


def test_get_form_action_if_not_in_forms():
    form_action_name = "my_business_logic"
    domain = Domain.from_yaml(
        textwrap.dedent(
            """
    actions:
    - my_action
    """
        )
    )

    with pytest.raises(ActionNotFoundException):
        assert not action.action_for_name_or_text(form_action_name, domain, None)


@pytest.mark.parametrize(
    "end_to_end_utterance", ["Hi", f"{UTTER_PREFIX} is a dangerous start"]
)
def test_get_end_to_end_utterance_action(end_to_end_utterance: Text):
    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
    actions:
    - my_action
    {KEY_E2E_ACTIONS}:
    - {end_to_end_utterance}
    - Bye Bye
"""
        )
    )

    actual = action.action_for_name_or_text(end_to_end_utterance, domain, None)

    assert isinstance(actual, ActionEndToEndResponse)
    assert actual.name() == end_to_end_utterance


async def test_run_end_to_end_utterance_action():
    end_to_end_utterance = "Hi"

    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
    actions:
    - my_action
    {KEY_E2E_ACTIONS}:
    - {end_to_end_utterance}
    - Bye Bye
"""
        )
    )

    e2e_action = action.action_for_name_or_text("Hi", domain, None)
    events = await e2e_action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        DialogueStateTracker.from_events("sender", evts=[]),
        domain,
    )

    assert events == [
        BotUttered(
            end_to_end_utterance,
            {
                "elements": None,
                "quick_replies": None,
                "buttons": None,
                "attachment": None,
                "image": None,
                "custom": None,
            },
            {},
        )
    ]


@pytest.mark.parametrize(
    "user, slot_name, slot_value, new_user, updated_value",
    [
        (
            UserUttered(
                intent={"name": "inform"},
                entities=[{"entity": "city", "value": "London"}],
            ),
            "location",
            "London",
            UserUttered(
                intent={"name": "inform"},
                entities=[{"entity": "city", "value": "Berlin"}],
            ),
            "Berlin",
        ),
        (
            UserUttered(text="test@example.com", intent={"name": "inform"}),
            "email",
            "test@example.com",
            UserUttered(text="updated_test@example.com", intent={"name": "inform"}),
            "updated_test@example.com",
        ),
        (
            UserUttered(intent={"name": "affirm"}),
            "cancel_booking",
            True,
            UserUttered(intent={"name": "deny"}),
            False,
        ),
        (
            UserUttered(intent={"name": "deny"}),
            "cancel_booking",
            False,
            UserUttered(intent={"name": "affirm"}),
            True,
        ),
        (
            UserUttered(
                intent={"name": "inform"},
                entities=[
                    {"entity": "name", "value": "Bob"},
                    {"entity": "name", "value": "Mary"},
                ],
            ),
            "guest_names",
            ["Bob", "Mary"],
            UserUttered(
                intent={"name": "inform"},
                entities=[{"entity": "name", "value": "John"}],
            ),
            ["John"],
        ),
    ],
)
async def test_action_extract_slots_predefined_mappings(
    user: Event, slot_name: Text, slot_value: Any, new_user: Event, updated_value: Any
):
    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intents:
            - inform
            - greet
            - affirm
            - deny
            entities:
            - city
            - name
            slots:
              location:
                type: text
                influence_conversation: false
                mappings:
                - type: from_entity
                  entity: city
              email:
                type: text
                influence_conversation: false
                mappings:
                - type: from_text
                  intent: inform
                  not_intent: greet
              cancel_booking:
                type: bool
                influence_conversation: false
                mappings:
                - type: from_intent
                  intent: affirm
                  value: true
                - type: from_intent
                  intent: deny
                  value: false
              guest_names:
                type: list
                influence_conversation: false
                mappings:
                - type: from_entity
                  entity: name"""
        )
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)
    tracker = DialogueStateTracker.from_events("sender", evts=[user])

    with pytest.warns(None):
        events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )

    assert events == [SlotSet(slot_name, slot_value)]

    events.extend([user])
    tracker.update_with_events(events, domain)

    new_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert new_events == [SlotSet(slot_name, slot_value)]

    new_events.extend([BotUttered(), ActionExecuted("action_listen"), new_user])
    tracker.update_with_events(new_events, domain)

    updated_evts = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert updated_evts == [SlotSet(slot_name, updated_value)]


async def test_action_extract_slots_with_from_trigger_mappings():
    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intents:
            - greet
            - inform
            - register
            slots:
              email:
                type: text
                influence_conversation: false
                mappings:
                - type: from_text
                  intent: inform
                  not_intent: greet
              existing_customer:
                type: bool
                influence_conversation: false
                mappings:
                - type: from_trigger_intent
                  intent: register
                  value: false
            forms:
              registration_form:
                required_slots:
                  - email"""
        )
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)
    user_event = UserUttered(text="I'd like to register", intent={"name": "register"})
    tracker = DialogueStateTracker.from_events(
        "sender", evts=[user_event, ActiveLoop("registration_form")]
    )
    events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert events == [SlotSet("existing_customer", False)]


@pytest.mark.parametrize(
    "slot_mapping, expected_value",
    [
        (
            {"type": "from_entity", "entity": "some_slot", "intent": "greet"},
            "some_value",
        ),
        (
            {"type": "from_intent", "intent": "greet", "value": "other_value"},
            "other_value",
        ),
        ({"type": "from_text"}, "bla"),
        ({"type": "from_text", "intent": "greet"}, "bla"),
        ({"type": "from_text", "not_intent": "other"}, "bla"),
    ],
)
async def test_action_extract_slots_when_mapping_applies(
    slot_mapping: Dict, expected_value: Text
):
    form_name = "test_form"
    entity_name = "some_slot"

    domain = Domain.from_dict(
        {
            "intents": ["greet"],
            "entities": ["some_slot"],
            "slots": {entity_name: {"type": "text", "mappings": [slot_mapping]}},
            "forms": {form_name: {REQUIRED_SLOTS_KEY: [entity_name]}},
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla",
                intent={"name": "greet", "confidence": 1.0},
                entities=[{"entity": entity_name, "value": "some_value"}],
            ),
        ],
    )
    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    # check that the value was extracted for correct intent
    assert slot_events == [SlotSet("some_slot", expected_value)]


@pytest.mark.parametrize(
    "entities, expected_slot_values",
    [
        # Two entities were extracted for `ListSlot`
        (
            [
                {"entity": "topping", "value": "mushrooms"},
                {"entity": "topping", "value": "kebab"},
            ],
            ["mushrooms", "kebab"],
        ),
        # Only one entity was extracted for `ListSlot`
        ([{"entity": "topping", "value": "kebab"}], ["kebab"]),
    ],
)
async def test_action_extract_slots_with_list_slot(
    entities: List[Dict[Text, Any]], expected_slot_values: List[Text]
):
    form_name = "order_form"
    slot_name = "toppings"

    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

    entities:
    - topping

    slots:
      {slot_name}:
        type: list
        influence_conversation: false
        mappings:
        - type: from_entity
          entity: topping

    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
          - {slot_name}
    """
        )
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, slot_name),
            UserUttered(
                "bla", intent={"name": "greet", "confidence": 1.0}, entities=entities
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        slots=domain.slots,
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert slot_events == [SlotSet(slot_name, expected_slot_values)]


@pytest.mark.parametrize(
    "slot_mapping",
    [
        {"type": "from_entity", "entity": "some_slot", "intent": "some_intent"},
        {"type": "from_intent", "intent": "some_intent", "value": "some_value"},
        {"type": "from_intent", "intent": "greeted", "value": "some_value"},
        {"type": "from_text", "intent": "other"},
        {"type": "from_text", "not_intent": "greet"},
        {"type": "from_trigger_intent", "intent": "some_intent", "value": "value"},
    ],
)
async def test_action_extract_slots_mapping_does_not_apply(slot_mapping: Dict):
    form_name = "some_form"
    entity_name = "some_slot"

    domain = Domain.from_dict(
        {
            "slots": {entity_name: {"type": "text", "mappings": [slot_mapping]}},
            "forms": {form_name: {REQUIRED_SLOTS_KEY: [entity_name]}},
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            UserUttered(
                "bla",
                intent={"name": "greet", "confidence": 1.0},
                entities=[{"entity": entity_name, "value": "some_value"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    # check that the value was not extracted for incorrect intent
    assert slot_events == []


async def test_action_extract_slots_with_matched_mapping_condition():
    form_name = "some_form"

    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intent:
            - greet
            - inform
            slots:
              name:
                type: text
                influence_conversation: true
                mappings:
                - type: from_text
                  conditions:
                  - active_loop: some_form
                    requested_slot: name
                  - active_loop: other_form
            forms:
             {form_name}:
               required_slots:
                 - name
             other_form:
               required_slots:
                 - name
            """
        )
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "name"),
            UserUttered(
                "Emily", intent={"name": "inform", "confidence": 1.0}, entities=[]
            ),
        ],
    )
    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert slot_events == [SlotSet("name", "Emily")]


async def test_action_extract_slots_no_matched_mapping_conditions():
    form_name = "some_form"

    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intent:
            - greet
            - inform
            entities:
            - email
            - name
            slots:
              name:
                type: text
                influence_conversation: false
                mappings:
                - type: from_entity
                  entity: name
                  conditions:
                  - active_loop: some_form
                    requested_slot: email
              email:
                type: text
                influence_conversation: false
                mappings:
                - type: from_entity
                  entity: email
            forms:
             {form_name}:
               required_slots:
                 - email
                 - name
            """
        )
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "name"),
            UserUttered(
                "My name is Emily.",
                intent={"name": "inform", "confidence": 1.0},
                entities=[{"entity": "name", "value": "Emily"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert slot_events == []


@pytest.mark.parametrize(
    "mapping_not_intent, mapping_intent, mapping_role, "
    "mapping_group, entities, intent, expected_slot_events",
    [
        (
            "some_intent",
            None,
            None,
            None,
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            [],
        ),
        (
            None,
            "some_intent",
            None,
            None,
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            [SlotSet("some_slot", "some_value")],
        ),
        (
            "some_intent",
            None,
            None,
            None,
            [{"entity": "some_entity", "value": "some_value"}],
            "some_other_intent",
            [SlotSet("some_slot", "some_value")],
        ),
        (
            None,
            None,
            "some_role",
            None,
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            [],
        ),
        (
            None,
            None,
            "some_role",
            None,
            [{"entity": "some_entity", "value": "some_value", "role": "some_role"}],
            "some_intent",
            [SlotSet("some_slot", "some_value")],
        ),
        (
            None,
            None,
            None,
            "some_group",
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            [],
        ),
        (
            None,
            None,
            None,
            "some_group",
            [{"entity": "some_entity", "value": "some_value", "group": "some_group"}],
            "some_intent",
            [SlotSet("some_slot", "some_value")],
        ),
        (
            None,
            None,
            "some_role",
            "some_group",
            [
                {
                    "entity": "some_entity",
                    "value": "some_value",
                    "group": "some_group",
                    "role": "some_role",
                }
            ],
            "some_intent",
            [SlotSet("some_slot", "some_value")],
        ),
        (
            None,
            None,
            "some_role",
            "some_group",
            [{"entity": "some_entity", "value": "some_value", "role": "some_role"}],
            "some_intent",
            [],
        ),
        (
            None,
            None,
            None,
            None,
            [
                {
                    "entity": "some_entity",
                    "value": "some_value",
                    "group": "some_group",
                    "role": "some_role",
                }
            ],
            "some_intent",
            # nothing should be extracted, because entity contain role and group
            # but mapping expects them to be None
            [],
        ),
    ],
)
async def test_action_extract_slots_from_entity(
    mapping_not_intent: Optional[Text],
    mapping_intent: Optional[Text],
    mapping_role: Optional[Text],
    mapping_group: Optional[Text],
    entities: List[Dict[Text, Any]],
    intent: Text,
    expected_slot_events: List[SlotSet],
):
    """Test extraction of a slot value from entity with the different restrictions."""
    form_name = "some form"
    form = FormAction(form_name, None)

    mapping = form.from_entity(
        entity="some_entity",
        role=mapping_role,
        group=mapping_group,
        intent=mapping_intent,
        not_intent=mapping_not_intent,
    )
    domain = Domain.from_dict(
        {
            "entities": ["some_entity"],
            "slots": {"some_slot": {"type": "any", "mappings": [mapping]}},
            "forms": {form_name: {REQUIRED_SLOTS_KEY: ["some_slot"]}},
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla", intent={"name": intent, "confidence": 1.0}, entities=entities
            ),
        ],
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert slot_events == expected_slot_events


@pytest.mark.parametrize(
    "entities, expected_slot_values",
    [
        # Two entities were extracted for `ListSlot`
        (
            [
                {"entity": "topping", "value": "mushrooms"},
                {"entity": "topping", "value": "kebab"},
            ],
            ["mushrooms", "kebab"],
        ),
        # Only one entity was extracted for `ListSlot`
        ([{"entity": "topping", "value": "kebab"}], ["kebab"]),
    ],
)
async def test_extract_other_list_slot_from_entity(
    entities: List[Dict[Text, Any]], expected_slot_values: List[Text]
):
    form_name = "some_form"
    slot_name = "toppings"
    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

    entities:
    - topping
    - some_slot

    intents:
    - some_intent
    - greeted

    slots:
      {slot_name}:
        type: list
        influence_conversation: false
        mappings:
        - type: from_entity
          entity: topping

    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
          - {slot_name}
    """
        )
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, "some slot"),
            UserUttered(
                "bla", intent={"name": "greet", "confidence": 1.0}, entities=entities
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        slots=domain.slots,
    )
    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert slot_events == [SlotSet(slot_name, expected_slot_values)]


@pytest.mark.parametrize(
    "trigger_slot_mapping, expected_value",
    [
        ({"type": "from_trigger_intent", "intent": "greet", "value": "ten"}, "ten"),
        (
            {
                "type": "from_trigger_intent",
                "intent": ["bye", "greet"],
                "value": "tada",
            },
            "tada",
        ),
    ],
)
async def test_trigger_slot_mapping_applies(
    trigger_slot_mapping: Dict, expected_value: Text
):
    form_name = "some_form"
    entity_name = "some_slot"
    slot_filled_by_trigger_mapping = "other_slot"

    domain = Domain.from_dict(
        {
            "slots": {
                entity_name: {
                    "type": "text",
                    "mappings": [
                        {
                            "type": "from_entity",
                            "entity": entity_name,
                            "intent": "some_intent",
                        }
                    ],
                },
                slot_filled_by_trigger_mapping: {
                    "type": "text",
                    "mappings": [trigger_slot_mapping],
                },
            },
            "forms": {form_name: {REQUIRED_SLOTS_KEY: [entity_name]}},
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla",
                intent={"name": "greet", "confidence": 1.0},
                entities=[{"entity": entity_name, "value": "some_value"}],
            ),
        ],
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert slot_events == [SlotSet(slot_filled_by_trigger_mapping, expected_value)]


@pytest.mark.parametrize(
    "trigger_slot_mapping",
    [
        ({"type": "from_trigger_intent", "intent": "bye", "value": "ten"}),
        ({"type": "from_trigger_intent", "not_intent": ["greet"], "value": "tada"}),
    ],
)
async def test_trigger_slot_mapping_does_not_apply(trigger_slot_mapping: Dict):
    form_name = "some_form"
    entity_name = "some_slot"
    slot_filled_by_trigger_mapping = "other_slot"

    domain = Domain.from_dict(
        {
            "slots": {
                entity_name: {
                    "type": "text",
                    "mappings": [
                        {
                            "type": "from_entity",
                            "entity": entity_name,
                            "intent": "some_intent",
                        }
                    ],
                },
                slot_filled_by_trigger_mapping: {
                    "type": "text",
                    "mappings": [trigger_slot_mapping],
                },
            },
            "forms": {
                form_name: {
                    REQUIRED_SLOTS_KEY: [entity_name, slot_filled_by_trigger_mapping]
                }
            },
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla",
                intent={"name": "greet", "confidence": 1.0},
                entities=[{"entity": entity_name, "value": "some_value"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert slot_events == []


@pytest.mark.parametrize(
    "event, validate_return_events, expected_events",
    [
        (
            UserUttered(
                intent={"name": "inform"},
                entities=[{"entity": "city", "value": "london"}],
            ),
            [{"event": "slot", "name": "from_entity_slot", "value": "London"}],
            [SlotSet("from_entity_slot", "London")],
        ),
        (
            UserUttered("hi", intent={"name": "greet"}),
            [{"event": "slot", "name": "from_text_slot", "value": "Hi"}],
            [SlotSet("from_text_slot", "Hi")],
        ),
        (
            UserUttered(intent={"name": "affirm"}),
            [{"event": "slot", "name": "from_intent_slot", "value": True}],
            [SlotSet("from_intent_slot", True)],
        ),
        (
            UserUttered(intent={"name": "chitchat"}),
            [{"event": "slot", "name": "custom_slot", "value": True}],
            [SlotSet("custom_slot", True)],
        ),
        (UserUttered("bla"), [], []),
    ],
)
async def test_action_extract_slots_execute_validation_action(
    event: Event,
    validate_return_events: List[Dict[Text, Any]],
    expected_events: List[Event],
):
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        intents:
        - greet
        - inform
        - affirm
        - chitchat

        entities:
        - city

        slots:
          from_entity_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: city
          from_text_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: from_text
              intent: greet
          from_intent_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: from_intent
              intent: affirm
              value: True
          custom_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: custom

        actions:
        - action_validate_slot_mappings
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=[event])

    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(action_server_url, payload={"events": validate_return_events})

        action_server = EndpointConfig(action_server_url)
        action_extract_slots = ActionExtractSlots(action_server)

        slot_events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert slot_events == expected_events


async def test_action_extract_slots_custom_action_and_predefined_slot_validation():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        intents:
        - inform

        entities:
        - city

        slots:
          location:
            type: text
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: city
          custom_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: custom
              action: action_test

        actions:
        - action_validate_slot_mappings
        - action_test
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered(
        intent={"name": "inform"}, entities=[{"entity": "city", "value": "london"}]
    )
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=[event])

    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [{"event": "slot", "name": "custom_slot", "value": "test"}]
            },
        )
        mocked.post(
            action_server_url,
            payload={
                "events": [{"event": "slot", "name": "location", "value": "London"}]
            },
        )

        action_server = EndpointConfig(action_server_url)
        action_extract_slots = ActionExtractSlots(action_server)

        slot_events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert slot_events == [
            SlotSet("location", "London"),
            SlotSet("custom_slot", "test"),
        ]


async def test_action_extract_slots_with_duplicate_custom_actions():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        intents:
        - inform

        entities:
        - city

        slots:
          custom_slot_one:
            type: float
            influence_conversation: false
            mappings:
            - type: custom
              action: action_test
          custom_slot_two:
            type: float
            influence_conversation: false
            mappings:
            - type: custom
              action: action_test

        actions:
        - action_test
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi")
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=[event])

    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "slot", "name": "custom_slot_one", "value": 1},
                    {"event": "slot", "name": "custom_slot_two", "value": 2},
                ]
            },
        )
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "slot", "name": "custom_slot_one", "value": 1},
                    {"event": "slot", "name": "custom_slot_two", "value": 2},
                ]
            },
        )

        action_server = EndpointConfig(action_server_url)
        action_extract_slots = ActionExtractSlots(action_server)

        slot_events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )

        assert len(mocked.requests) == 1

        assert len(slot_events) == 2
        assert SlotSet("custom_slot_two", 2) in slot_events
        assert SlotSet("custom_slot_one", 1) in slot_events


async def test_action_extract_slots_disallowed_events(caplog: LogCaptureFixture):
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        slots:
          custom_slot_one:
            type: float
            influence_conversation: false
            mappings:
            - type: custom
              action: action_test

        actions:
        - action_test
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi")
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=[event])

    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "slot", "name": "custom_slot_one", "value": 1},
                    {"event": "reset_slots"},
                ]
            },
        )

        action_server = EndpointConfig(action_server_url)
        action_extract_slots = ActionExtractSlots(action_server)

        with caplog.at_level(logging.INFO):
            slot_events = await action_extract_slots.run(
                CollectingOutputChannel(),
                TemplatedNaturalLanguageGenerator(domain.responses),
                tracker,
                domain,
            )

        caplog_info_records = list(
            filter(lambda x: x[1] == logging.INFO, caplog.record_tuples)
        )

        assert all(
            [
                "Running custom action 'action_test' has resulted "
                "in an event of type 'reset_slots'." in record[2]
                for record in caplog_info_records
            ]
        )
        assert slot_events == [SlotSet("custom_slot_one", 1)]


@pytest.mark.parametrize(
    "exception",
    [
        RasaException("Test"),
        ClientResponseError(400, "Test", '{"action_name": "action_test"}'),
    ],
)
async def test_action_extract_slots_warns_custom_action_exceptions(
    caplog: LogCaptureFixture, exception: Exception
):
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        slots:
          custom_slot_one:
            type: float
            influence_conversation: false
            mappings:
            - type: custom
              action: action_test

        actions:
        - action_test
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi")
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=[event])

    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(action_server_url, exception=exception)

        action_server = EndpointConfig(action_server_url)
        action_extract_slots = ActionExtractSlots(action_server)

        with caplog.at_level(logging.WARNING):
            await action_extract_slots.run(
                CollectingOutputChannel(),
                TemplatedNaturalLanguageGenerator(domain.responses),
                tracker,
                domain,
            )

        assert any(
            [
                "The default action 'action_extract_slots' failed to fill "
                "slots with custom mappings." in message
                for message in caplog.messages
            ]
        )


async def test_action_extract_slots_with_empty_conditions():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        entities:
        - city

        slots:
          location:
            type: float
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: city
              conditions: []
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi", entities=[{"entity": "city", "value": "Berlin"}])
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=[event])

    action_extract_slots = ActionExtractSlots(None)

    with pytest.warns(None):
        events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
    assert events == [SlotSet("location", "Berlin")]


async def test_action_extract_slots_with_not_existing_entity():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        entities:
        - city

        slots:
          location:
            type: float
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: city2
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi", entities=[{"entity": "city", "value": "Berlin"}])
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=[event])

    action_extract_slots = ActionExtractSlots(None)

    with pytest.warns(
        None,
        match="Slot 'location' uses a `from_entity` mapping for "
        "a non-existent entity 'city2'. "
        "Skipping slot extraction because of invalid mapping.",
    ):
        events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
    assert events == []


async def test_action_extract_slots_with_not_existing_intent():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        intents:
        - greet

        slots:
          location:
            type: text
            influence_conversation: false
            mappings:
            - type: from_intent
              intent: affirm
              value: some_value
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi", entities=[{"entity": "city", "value": "Berlin"}])
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=[event])

    action_extract_slots = ActionExtractSlots(None)

    with pytest.warns(
        UserWarning,
        match=r"Slot 'location' uses a 'from_intent' mapping for "
        r"a non-existent intent 'affirm'. "
        r"Skipping slot extraction because of invalid mapping.",
    ):
        events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
    assert events == []


async def test_action_extract_slots_with_none_value_predefined_mapping():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        entities:
        - some_entity

        slots:
          some_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: some_entity

          custom_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: custom

        actions:
        - action_validate_slot_mappings
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi", entities=[{"entity": "some_entity", "value": None}])
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=[event])

    action_extract_slots = ActionExtractSlots(None)

    events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert events == []


async def test_action_extract_slots_with_none_value_custom_mapping():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        slots:
          custom_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: custom

        actions:
        - action_validate_slot_mappings
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi")
    tracker = DialogueStateTracker.from_events(
        sender_id="test_id", evts=[event], slots=domain.slots
    )

    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [{"event": "slot", "name": "custom_slot", "value": None}]
            },
        )

        action_server = EndpointConfig(action_server_url)
        action_extract_slots = ActionExtractSlots(action_server)
        events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert events == [SlotSet("custom_slot", None)]


async def test_action_extract_slots_returns_bot_uttered():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        slots:
          custom_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: custom

        actions:
        - action_validate_slot_mappings
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi")
    tracker = DialogueStateTracker.from_events(
        sender_id="test_id", evts=[event], slots=domain.slots
    )

    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "slot", "name": "custom_slot", "value": "test"},
                    {"event": "bot", "text": "Information recorded."},
                ]
            },
        )

        action_server = EndpointConfig(action_server_url)
        action_extract_slots = ActionExtractSlots(action_server)
        events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert all([isinstance(event, (SlotSet, BotUttered)) for event in events])


async def test_action_extract_slots_does_not_raise_disallowed_warning_for_slot_events(
    caplog: LogCaptureFixture,
):
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        slots:
          custom_slot_a:
            type: text
            influence_conversation: false
            mappings:
            - type: custom
              action: custom_extract_action
          custom_slot_b:
            type: text
            influence_conversation: false
            mappings:
            - type: custom

        actions:
        - custom_extract_action
        - action_validate_slot_mappings
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi")
    tracker = DialogueStateTracker.from_events(
        sender_id="test_id", evts=[event], slots=domain.slots
    )

    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "slot", "name": "custom_slot_a", "value": "test_A"}
                ]
            },
        )

        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "slot", "name": "custom_slot_b", "value": "test_B"}
                ]
            },
        )

        action_server = EndpointConfig(action_server_url)
        action_extract_slots = ActionExtractSlots(action_server)

        with caplog.at_level(logging.INFO):
            events = await action_extract_slots.run(
                CollectingOutputChannel(),
                TemplatedNaturalLanguageGenerator(domain.responses),
                tracker,
                domain,
            )

        caplog_info_records = list(
            filter(lambda x: x[1] == logging.INFO, caplog.record_tuples)
        )
        assert len(caplog_info_records) == 0

        assert events == [
            SlotSet("custom_slot_b", "test_B"),
            SlotSet("custom_slot_a", "test_A"),
        ]


async def test_action_extract_slots_non_required_form_slot_with_from_entity_mapping():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        intents:
        - form_start
        - intent1
        - intent2

        entities:
        - form1_info1
        - form1_slot1
        - form1_slot2

        slots:
          form1_info1:
            type: text
            mappings:
            - type: from_entity
              entity: form1_info1

          form1_slot1:
            type: text
            influence_conversation: false
            mappings:
            - type: from_intent
              value: Filled
              intent: intent1
              conditions:
              - active_loop: form1
                requested_slot: form1_slot1

          form1_slot2:
            type: text
            influence_conversation: false
            mappings:
            - type: from_intent
              value: Filled
              intent: intent2
              conditions:
              - active_loop: form1
                requested_slot: form1_slot2
        forms:
          form1:
            required_slots:
            - form1_slot1
            - form1_slot2
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    initial_events = [
        UserUttered("Start form."),
        ActiveLoop("form1"),
        SlotSet(REQUESTED_SLOT, "form1_slot1"),
        UserUttered(
            "Hi",
            intent={"name": "intent1"},
            entities=[{"entity": "form1_info1", "value": "info1"}],
        ),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=initial_events)

    action_extract_slots = ActionExtractSlots(None)

    events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert events == [SlotSet("form1_info1", "info1"), SlotSet("form1_slot1", "Filled")]


@pytest.mark.parametrize(
    "starting_value, value_to_set, expect_event",
    [
        ("value", None, True),
        (None, "value", True),
        ("value", "value", True),
        (None, None, False),
    ],
)
async def test_action_extract_slots_emits_necessary_slot_set_events(
    starting_value: Text, value_to_set: Text, expect_event: bool
):
    entity_name = "entity"
    intent_name = "intent_with_entity"

    domain = textwrap.dedent(
        f"""
          intents:
            - {intent_name}
          entities:
            - {entity_name}
          slots:
            {entity_name}:
              type: text
              mappings:
              - type: from_entity
                entity: {entity_name}
        """
    )

    domain = Domain.from_yaml(domain)

    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            SlotSet(entity_name, starting_value),
        ],
    )

    tracker.update_with_events(
        new_events=[
            UserUttered(
                text="I am a text",
                intent={"name": intent_name},
                entities=[{"entity": entity_name, "value": value_to_set}],
            )
        ],
        domain=domain,
    )

    action = ActionExtractSlots(None)

    events = await action.run(
        output_channel=CollectingOutputChannel(),
        nlg=Mock(),
        tracker=tracker,
        domain=domain,
    )

    if expect_event:
        assert len(events) == 1
        assert type(events[0]) == SlotSet
        assert events[0].key == entity_name
        assert events[0].value == value_to_set
    else:
        assert len(events) == 0


async def test_action_extract_slots_priority_of_slot_mappings():
    slot_name = "location_slot"
    entity_name = "location"
    entity_value = "Berlin"

    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        intents:
        - inform

        entities:
        - {entity_name}

        slots:
          {slot_name}:
            type: text
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: {entity_name}
            - type: from_intent
              value: 42
              intent: inform

        responses:
            utter_ask_location:
                - text: "where are you located?"
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    initial_events = [
        UserUttered(
            "I am located in Berlin",
            intent={"name": "inform"},
            entities=[{"entity": entity_name, "value": entity_value}],
        ),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=initial_events)

    action_extract_slots = ActionExtractSlots(None)

    events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    tracker.update_with_events(events, domain=domain)
    assert tracker.get_slot("location_slot") == entity_value


async def test_action_extract_slots_allows_slotset_for_same_value(
    caplog: LogCaptureFixture,
):
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        slots:
          custom_slot_a:
            type: text
            influence_conversation: false
            mappings:
            - type: custom
              action: custom_extract_action

        actions:
        - custom_extract_action
        - action_validate_slot_mappings
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    event = UserUttered("Hi")
    tracker = DialogueStateTracker.from_events(
        sender_id="test_id", evts=[event], slots=domain.slots
    )

    # Set the value of the slot in the tracker manually
    tracker.update(SlotSet("custom_slot_a", "test_A"))
    action_server_url = "https://my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "slot", "name": "custom_slot_a", "value": "test_A"}
                ]
            },
        )

        action_server = EndpointConfig(action_server_url)
        action_extract_slots = ActionExtractSlots(action_server)

        with caplog.at_level(logging.INFO):
            events = await action_extract_slots.run(
                CollectingOutputChannel(),
                TemplatedNaturalLanguageGenerator(domain.responses),
                tracker,
                domain,
            )

        caplog_info_records = list(
            filter(lambda x: x[1] == logging.INFO, caplog.record_tuples)
        )
        assert len(caplog_info_records) == 0
        assert events == [SlotSet("custom_slot_a", "test_A")]


async def test_action_extract_slots_active_loop_none_in_mapping_condition():
    entity = "name"
    entity_value = "Julia"
    slot = "user_name"

    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        intents:
        - greet

        entities:
        - {entity}

        slots:
          {slot}:
            type: text
            mappings:
            - type: from_entity
              entity: {entity}
              conditions:
              - active_loop: null
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    initial_events = [
        UserUttered(
            "Hi, I'm Julia.",
            intent={"name": "greet"},
            entities=[{"entity": entity, "value": entity_value}],
        ),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=initial_events)

    action_extract_slots = ActionExtractSlots(None)

    events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert events == [SlotSet(slot, entity_value)]


async def test_action_extract_slots_active_loop_none_does_not_set_slot_in_form():
    entity = "name"
    entity_value = "Julia"
    slot = "user_name"

    domain_yaml = textwrap.dedent(
        f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

            intents:
            - greet

            entities:
            - {entity}

            slots:
              {slot}:
                type: text
                mappings:
                - type: from_entity
                  entity: {entity}
                  conditions:
                  - active_loop: null

            forms:
              my_form:
                required_slots: []
            """
    )
    domain = Domain.from_yaml(domain_yaml)
    initial_events = [
        ActiveLoop("my_form"),
        UserUttered(
            "Hi, I'm Julia.",
            intent={"name": "greet"},
            entities=[{"entity": entity, "value": entity_value}],
        ),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="test_id", evts=initial_events)

    action_extract_slots = ActionExtractSlots(None)

    events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert events == []
