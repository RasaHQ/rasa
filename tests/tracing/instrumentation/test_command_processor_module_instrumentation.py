import importlib
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.dialogue_understanding.commands.correct_slots_command import (
    CorrectSlotsCommand,
    CorrectedSlot,
)
from rasa.shared.core.events import UserUttered, DialogueStackUpdated
from rasa.shared.core.flows import FlowsList, Flow
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.instrumentation.instrumentation import COMMAND_PROCESSOR_MODULE_NAME


def test_tracing_command_processor_execute_commands(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    sender_id = "test"
    instrumentation.instrument(tracer_provider)

    module = importlib.import_module(COMMAND_PROCESSOR_MODULE_NAME)
    tracker = DialogueStateTracker.from_events(sender_id, evts=[])
    module.execute_commands(
        tracker,
        FlowsList(underlying_flows=[]),
    )

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans == 4

    current_captured_spans = captured_spans[-num_captured_spans:]

    expected_span_names = [
        "command_processor.clean_up_commands",
        "command_processor.validate_state_of_commands",
        "command_processor.remove_duplicated_set_slots",
        "command_processor.execute_commands",
    ]

    for span, expected_name in zip(current_captured_spans, expected_span_names):
        assert span.name == expected_name

    # the parent span is always the last because it ends only after
    # all the children spans are finished
    parent_span = captured_spans[-1]

    expected_attributes = {
        "number_of_events": 0,
        "sender_id": sender_id,
    }
    assert parent_span.attributes == expected_attributes


def test_tracing_command_processor_clean_up_commands(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    # arrange
    sender_id = "test"
    user_event = UserUttered(
        "I want to make a transfer to Anna",
        parse_data={
            "commands": [
                {"command": "start flow", "flow": "transfer_money"},
                {"command": "set slot", "name": "recipient", "value": "Anna"},
            ]
        },
    )
    instrumentation.instrument(tracer_provider)

    module = importlib.import_module(COMMAND_PROCESSOR_MODULE_NAME)
    tracker = DialogueStateTracker.from_events(sender_id, evts=[user_event])
    commands = module.get_commands_from_tracker(tracker)

    # act
    module.clean_up_commands(
        commands,
        tracker,
        FlowsList(underlying_flows=[Flow("transfer_money")]),
    )

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    expected_values = [
        {"flow": "transfer_money", "command": "start flow"},
        {"name": "recipient", "command": "set slot"},
    ]

    assert captured_span.attributes == {"commands": str(expected_values)}


def test_tracing_command_processor_validate_state_of_commands(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    # arrange
    instrumentation.instrument(tracer_provider)

    module = importlib.import_module(COMMAND_PROCESSOR_MODULE_NAME)
    cleaned_up_commands = [
        CorrectSlotsCommand(corrected_slots=[CorrectedSlot("recipient", "John")])
    ]

    # act
    module.validate_state_of_commands(cleaned_up_commands)

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "command_processor.validate_state_of_commands"

    expected_attributes = {
        "cleaned_up_commands": "[{'corrected_slots': [{'name': 'recipient'}], 'command': 'correct slot'}]"  # noqa: E501
    }
    assert captured_span.attributes == expected_attributes


def test_tracing_command_processor_remove_duplicated_set_slots(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    # arrange
    instrumentation.instrument(tracer_provider)

    module = importlib.import_module(COMMAND_PROCESSOR_MODULE_NAME)
    events = [
        DialogueStackUpdated(
            update='[{"op": "add", "path": "/2", "value": {"frame_id": "H70JZZK1", "flow_id": "pattern_correction", "step_id": "START", "is_reset_only": false, "corrected_slots": {"transfer_money_recipient": "john"}, "reset_flow_id": "transfer_money", "reset_step_id": "0_collect_transfer_money_recipient", "type": "pattern_correction"}}]',  # noqa: E501
        )
    ]

    # act
    module.remove_duplicated_set_slots(events)

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "command_processor.remove_duplicated_set_slots"

    captured_attributes = captured_span.attributes
    assert "resulting_events" in captured_attributes
    assert "corrected_slots" not in captured_attributes["resulting_events"]
