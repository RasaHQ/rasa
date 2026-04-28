import uuid
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.actions.action_default_capabilities import (
    _NO_CAPABILITIES_TEXT,
    ActionDefaultCapabilities,
    _maybe_rephrase,
    _render_capabilities,
)
from rasa.core.channels import OutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.constants import ACTION_DEFAULT_CAPABILITIES_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig


def _make_flows(*flows: Flow) -> FlowsList:
    return FlowsList(list(flows))


@pytest.fixture
def sender_id() -> str:
    return uuid.uuid4().hex


@pytest.fixture
def test_tracker(sender_id: str) -> DialogueStateTracker:
    return DialogueStateTracker.from_events(
        sender_id=sender_id,
        evts=[],
        slots=[TextSlot("name", mappings=[])],
    )


def test_name() -> None:
    assert ActionDefaultCapabilities().name() == ACTION_DEFAULT_CAPABILITIES_NAME


def test_render_capabilities_with_and_without_description() -> None:
    rendered = _render_capabilities(
        [
            {"name": "Flow A", "description": "Description A"},
            {"name": "Flow B", "description": None},
        ]
    )

    assert (
        rendered
        == "Here's what I can help you with:\n- Flow A: Description A\n- Flow B"
    )


async def test_maybe_rephrase_returns_original_for_non_rephraser_nlg(
    test_tracker: DialogueStateTracker, default_nlg: TemplatedNaturalLanguageGenerator
) -> None:
    response = {"text": "Original text"}
    output_channel = MagicMock()

    result = await _maybe_rephrase(response, default_nlg, test_tracker, output_channel)

    assert result == response


async def test_maybe_rephrase_calls_contextual_rephraser(
    test_tracker: DialogueStateTracker,
    default_channel: OutputChannel,
    monkeypatch: MonkeyPatch,
) -> None:
    response = {"text": "Original text"}

    expected_rephrased_response = {"text": "Rephrased text"}
    mock_rephrase = AsyncMock(return_value=expected_rephrased_response)
    monkeypatch.setattr(ContextualResponseRephraser, "rephrase", mock_rephrase)
    monkeypatch.setattr(
        ContextualResponseRephraser, "perform_llm_health_check", Mock(return_value=None)
    )
    rephraser = ContextualResponseRephraser(
        EndpointConfig.from_dict({}), domain=Domain.empty()
    )

    result = await _maybe_rephrase(response, rephraser, test_tracker, default_channel)

    assert result == expected_rephrased_response
    mock_rephrase.assert_awaited_once_with(response, test_tracker, default_channel)


async def test_run_returns_fallback_when_flows_missing(
    test_tracker: DialogueStateTracker,
    default_nlg: TemplatedNaturalLanguageGenerator,
    default_channel: OutputChannel,
) -> None:
    action = ActionDefaultCapabilities()

    events = await action.run(
        output_channel=default_channel,
        nlg=default_nlg,
        tracker=test_tracker,
        domain=Domain.empty(),
        metadata=None,
        flows=None,
    )

    assert len(events) == 1
    assert events[0].text == _NO_CAPABILITIES_TEXT


async def test_run_returns_fallback_when_no_startable_flows(
    test_tracker: DialogueStateTracker,
    default_nlg: TemplatedNaturalLanguageGenerator,
    default_channel: OutputChannel,
) -> None:
    action = ActionDefaultCapabilities()
    flows = _make_flows(
        Flow(
            id="guarded",
            custom_name="Guarded Flow",
            description="Needs name",
            guard_condition="slots.name != null",
        )
    )

    events = await action.run(
        output_channel=default_channel,
        nlg=default_nlg,
        tracker=test_tracker,
        domain=Domain.empty(),
        flows=flows,
    )

    assert len(events) == 1
    assert events[0].text == _NO_CAPABILITIES_TEXT


async def test_run_renders_startable_flows_and_uses_id_when_name_missing(
    test_tracker: DialogueStateTracker,
    default_nlg: TemplatedNaturalLanguageGenerator,
    default_channel: OutputChannel,
) -> None:
    action = ActionDefaultCapabilities()
    test_tracker.update(SlotSet("name", "Alice"))
    flows = _make_flows(
        Flow(
            id="named_flow", custom_name="Named Flow", description="Named description"
        ),
        Flow(id="id_only_flow", description=None),
    )

    events = await action.run(
        output_channel=default_channel,
        nlg=default_nlg,
        tracker=test_tracker,
        domain=Domain.empty(),
        flows=flows,
    )

    assert len(events) == 1
    assert events[0].text == (
        "Here's what I can help you with:\n"
        "- Named Flow: Named description\n"
        "- id_only_flow"
    )


async def test_run_excludes_active_flow_from_rendered_capabilities(
    test_tracker: DialogueStateTracker,
    default_nlg: TemplatedNaturalLanguageGenerator,
    default_channel: OutputChannel,
) -> None:
    action = ActionDefaultCapabilities()
    test_tracker.update(SlotSet("name", "Alice"))
    test_tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "flow_id": "active_flow",
                    "step_id": "some_step",
                    "frame_id": "active-flow-frame",
                    "frame_type": "regular",
                    "type": "flow",
                }
            ]
        )
    )

    flows = _make_flows(
        Flow(id="active_flow", custom_name="Active Flow", description="Current flow"),
        Flow(
            id="other_flow", custom_name="Other Flow", description="Other description"
        ),
    )

    events = await action.run(
        output_channel=default_channel,
        nlg=default_nlg,
        tracker=test_tracker,
        domain=Domain.empty(),
        flows=flows,
    )

    assert len(events) == 1
    assert (
        events[0].text
        == "Here's what I can help you with:\n- Other Flow: Other description"
    )


async def test_run_excludes_flow_that_triggered_default_capabilities_action(
    test_tracker: DialogueStateTracker,
    default_nlg: TemplatedNaturalLanguageGenerator,
    default_channel: OutputChannel,
) -> None:
    """The helper flow containing this action should not advertise itself."""
    action = ActionDefaultCapabilities()
    test_tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "flow_id": "capabilities_help",
                    "step_id": "utter_capabilities",
                    "frame_id": "capabilities-help-frame",
                    "frame_type": "regular",
                    "type": "flow",
                }
            ]
        )
    )

    capabilities_help_flow = Flow.from_json(
        "capabilities_help",
        {
            "name": "Capabilities Help",
            "description": "Explains what the assistant can do.",
            "steps": [
                {
                    "id": "utter_capabilities",
                    "action": ACTION_DEFAULT_CAPABILITIES_NAME,
                }
            ],
        },
    )
    book_trip_flow = Flow(
        id="book_trip",
        custom_name="Book Trip",
        description="Helps users book a trip.",
    )

    events = await action.run(
        output_channel=default_channel,
        nlg=default_nlg,
        tracker=test_tracker,
        domain=Domain.empty(),
        flows=_make_flows(capabilities_help_flow, book_trip_flow),
    )

    assert len(events) == 1
    assert events[0].text == (
        "Here's what I can help you with:\n- Book Trip: Helps users book a trip."
    )
    assert "Capabilities Help" not in events[0].text
    assert "Explains what the assistant can do." not in events[0].text


async def test_run_returns_fallback_when_no_valid_flow(
    test_tracker: DialogueStateTracker,
    default_nlg: TemplatedNaturalLanguageGenerator,
    default_channel: OutputChannel,
) -> None:
    """Return fallback utterance when only capabilities flow is defined."""
    action = ActionDefaultCapabilities()
    test_tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "flow_id": "capabilities_help",
                    "step_id": "utter_capabilities",
                    "frame_id": "capabilities-help-frame",
                    "frame_type": "regular",
                    "type": "flow",
                }
            ]
        )
    )

    capabilities_help_flow = Flow.from_json(
        "capabilities_help",
        {
            "name": "Capabilities Help",
            "description": "Explains what the assistant can do.",
            "steps": [
                {
                    "id": "utter_capabilities",
                    "action": ACTION_DEFAULT_CAPABILITIES_NAME,
                }
            ],
        },
    )

    events = await action.run(
        output_channel=default_channel,
        nlg=default_nlg,
        tracker=test_tracker,
        domain=Domain.empty(),
        flows=_make_flows(
            capabilities_help_flow,
        ),
    )

    assert len(events) == 1
    assert events[0].text == _NO_CAPABILITIES_TEXT
