import importlib
from typing import List, Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.events import SessionStarted
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.steps import CollectInformationFlowStep
from rasa.shared.core.flows.steps.constants import START_STEP
from rasa.shared.core.slots import Slot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.instrumentation.instrumentation import FLOW_EXECUTOR_MODULE_NAME


@pytest.fixture
def available_actions() -> List[str]:
    return ["action_listen", "utter_ask_transfer_money_amount_of_money"]


@pytest.fixture
def tracker() -> DialogueStateTracker:
    return DialogueStateTracker.from_events("test", evts=[SessionStarted()])


@pytest.fixture
def slots() -> List[Slot]:
    return []


@pytest.fixture
def dialogue_stack() -> DialogueStack:
    return DialogueStack(frames=[])


@pytest.fixture
def flow_step() -> CollectInformationFlowStep:
    return CollectInformationFlowStep(
        idx=1,
        custom_id="ask_amount",
        description="ask for the amount of money to transfer",
        metadata={"channel": "socketio"},
        next=FlowStepLinks(links=[]),
        collect="amount",
        utter="utter_ask_transfer_money_amount_of_money",
        collect_action="action_ask_transfer_money_amount_of_money",
        rejections=[],
        flow_id="transfer_money_test_flow",
    )


@pytest.fixture
def flow(flow_step: CollectInformationFlowStep) -> Flow:
    return Flow(
        id="transfer_money", step_sequence=FlowStepSequence(child_steps=[flow_step])
    )


@pytest.mark.asyncio
async def test_tracing_flow_executor_advance_flows(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    tracker: DialogueStateTracker,
    available_actions: List[str],
    flow: Flow,
) -> None:
    instrumentation.instrument(tracer_provider)

    module = importlib.import_module(FLOW_EXECUTOR_MODULE_NAME)
    await module.advance_flows(
        tracker,
        available_actions,
        FlowsList(underlying_flows=[flow]),
        slots=[],
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans == 1

    current_span = captured_spans[-1]
    assert current_span.name == "rasa.core.policies.flows.flow_executor.advance_flows"
    assert current_span.attributes == {
        "available_actions": '["action_listen", '
        '"utter_ask_transfer_money_amount_of_money"]',
        "current_context": "{}",
    }


@pytest.mark.asyncio
@pytest.mark.skip(reason="We need to update instrumentation. Fix this later.")
async def test_tracing_flow_executor_run_step(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    tracker: DialogueStateTracker,
    available_actions: List[str],
    flow_step: CollectInformationFlowStep,
    flow: Flow,
) -> None:
    instrumentation.instrument(tracer_provider)

    module = importlib.import_module(FLOW_EXECUTOR_MODULE_NAME)

    await module.run_step(
        flow_step,
        flow,
        tracker.stack,
        tracker,
        available_actions,
        FlowsList(underlying_flows=[flow]),
        previous_step_id=START_STEP,
        slots=[],
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans == 1

    current_span = captured_spans[-1]
    assert current_span.name == "CollectInformationFlowStep.run_step"

    expected_attributes = {
        "step_custom_id": "ask_amount",
        "step_description": "ask for the amount of money to transfer",
        "current_flow_id": "transfer_money",
        "current_context": "{}",
        "previous_step_id": START_STEP,
        "slots": "[]",
    }
    assert current_span.attributes == expected_attributes


@pytest.mark.asyncio
@pytest.mark.skip(reason="We need to update instrumentation. Fix this later.")
async def test_tracing_flow_executor_advance_flows_until_next_action(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    tracker: DialogueStateTracker,
    available_actions: List[str],
    flow: Flow,
) -> None:
    instrumentation.instrument(tracer_provider)

    module = importlib.import_module(FLOW_EXECUTOR_MODULE_NAME)

    await module.advance_flows_until_next_action(
        tracker, available_actions, FlowsList(underlying_flows=[flow]), []
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans == 1

    current_span = captured_spans[-1]

    assert (
        current_span.name
        == "rasa.core.policies.flows.flow_executor.advance_flows_until_next_action"
    )
    assert current_span.attributes == {
        "action_name": "action_listen",
        "score": 1.0,
        "metadata": '{"active_flow": null, "step_id": null}',
        "events": "[]",
    }
