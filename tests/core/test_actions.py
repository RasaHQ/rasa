import textwrap
from typing import List, Text

import pytest
from aioresponses import aioresponses

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
)
from rasa.core.actions.forms import FormAction
from rasa.core.channels import CollectingOutputChannel
from rasa.shared.constants import UTTER_PREFIX
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
)
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
    RULE_SNIPPET_ACTION_NAME,
    ACTIVE_LOOP,
    FOLLOWUP_ACTION,
    REQUESTED_SLOT,
    SESSION_START_METADATA_SLOT,
)
from rasa.shared.core.trackers import DialogueStateTracker
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
    )

    instantiated_actions = [
        action.action_for_name_or_text(action_name, domain, None)
        for action_name in domain.action_names_or_texts
    ]

    assert len(instantiated_actions) == 14
    assert instantiated_actions[0].name() == ACTION_LISTEN_NAME
    assert instantiated_actions[1].name() == ACTION_RESTART_NAME
    assert instantiated_actions[2].name() == ACTION_SESSION_START_NAME
    assert instantiated_actions[3].name() == ACTION_DEFAULT_FALLBACK_NAME
    assert instantiated_actions[4].name() == ACTION_DEACTIVATE_LOOP_NAME
    assert instantiated_actions[5].name() == ACTION_REVERT_FALLBACK_EVENTS_NAME
    assert instantiated_actions[6].name() == ACTION_DEFAULT_ASK_AFFIRMATION_NAME
    assert instantiated_actions[7].name() == ACTION_DEFAULT_ASK_REPHRASE_NAME
    assert instantiated_actions[8].name() == ACTION_TWO_STAGE_FALLBACK_NAME
    assert instantiated_actions[9].name() == ACTION_BACK_NAME
    assert instantiated_actions[10].name() == RULE_SNIPPET_ACTION_NAME
    assert instantiated_actions[11].name() == "my_module.ActionTest"
    assert instantiated_actions[12].name() == "utter_test"
    assert instantiated_actions[13].name() == "utter_chitchat"


async def test_remote_action_runs(
    default_channel, default_nlg, default_tracker, domain: Domain
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
    default_channel, default_nlg, default_tracker, domain: Domain
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
    default_channel, default_tracker, domain: Domain
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
    assert "Failed to execute custom action." in str(execinfo.value)


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
        assert "Failed to execute custom action." in str(execinfo.value)


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
    from rasa.core.channels.slack import SlackBot

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
            "congrats, you've restarted me!",
            metadata={"utter_action": "utter_restart"},
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
    "slot_mapping",
    [
        """
    my_slot:
    - type:from_text
    """,
        "",
    ],
)
def test_get_form_action(slot_mapping: Text):
    form_action_name = "my_business_logic"
    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
    actions:
    - my_action
    forms:
      {form_action_name}:
        {slot_mapping}
    """
        )
    )

    actual = action.action_for_name_or_text(form_action_name, domain, None)
    assert isinstance(actual, FormAction)


def test_get_form_action_with_rasa_open_source_1_forms():
    form_action_name = "my_business_logic"
    with pytest.warns(FutureWarning):
        domain = Domain.from_yaml(
            textwrap.dedent(
                f"""
        actions:
        - my_action
        forms:
        - {form_action_name}
        """
            )
        )

    actual = action.action_for_name_or_text(form_action_name, domain, None)
    assert isinstance(actual, RemoteAction)


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
