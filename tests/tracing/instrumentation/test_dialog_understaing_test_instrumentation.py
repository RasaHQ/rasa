import importlib
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from rasa.dialogue_understanding_test.command_metric_calculation import CommandMetrics
from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestSuiteResult,
)
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.instrumentation.instrumentation import (
    DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME,
)


def test_tracing_command_processor_execute_commands(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    instrumentation.instrument(tracer_provider)

    test_suite_results = DialogueUnderstandingTestSuiteResult()
    test_suite_results.number_of_passed_tests = 42
    test_suite_results.number_of_failed_tests = 3
    test_suite_results.command_metrics = {
        "start": CommandMetrics(tp=8, fp=2, fn=1, total_count=9)
    }

    module = importlib.import_module(DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME)
    module.print_test_results(test_suite_results, False)

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert (
        captured_span.name == "rasa.dialogue_understanding_test.io.print_test_results"
    )
    assert captured_span.attributes["number_of_passed_tests"] == 42
    assert captured_span.attributes["number_of_failed_tests"] == 3
    assert '"start":' in captured_span.attributes["command_metrics"]
    assert '"tp": 8' in captured_span.attributes["command_metrics"]
