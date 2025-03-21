import importlib
from typing import Any, Dict, Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from rasa.dialogue_understanding_test.command_metrics import CommandMetrics
from rasa.dialogue_understanding_test.du_test_result import (
    KEY_COMMANDS_F1_MACRO,
    KEY_COMMANDS_F1_MICRO,
    KEY_COMMANDS_F1_WEIGHTED,
    DialogueUnderstandingTestSuiteResult,
)
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.instrumentation.instrumentation import (
    DIALOG_UNDERSTANDING_TEST_IO_MODULE_NAME,
)


@pytest.mark.parametrize(
    "llm_config",
    [
        # config using model group syntax
        {
            "id": "openai-direct-gpt-4",
            "models": [
                {
                    "model": "gpt-4",
                    "provider": "openai",
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "timeout": 7,
                    "top_p": 0.0,
                }
            ],
        },
        # config using old model syntax
        {
            "model": "gpt-4",
            "provider": "openai",
            "temperature": 0.0,
            "max_tokens": 256,
            "timeout": 7,
            "top_p": 0.0,
        },
    ],
)
def test_dut_print_test_results_instrumentation(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    llm_config: Dict[str, Any],
) -> None:
    instrumentation.instrument(tracer_provider)

    test_suite_results = DialogueUnderstandingTestSuiteResult()
    test_suite_results.number_of_passed_tests = 42
    test_suite_results.number_of_failed_tests = 3
    test_suite_results.command_metrics = {
        "start": CommandMetrics(tp=8, fp=2, fn=1, total_count=9)
    }
    test_suite_results.f1_score[KEY_COMMANDS_F1_MACRO] = (
        DialogueUnderstandingTestSuiteResult.calculate_f1_macro(
            test_suite_results.command_metrics
        )
    )
    test_suite_results.f1_score[KEY_COMMANDS_F1_MICRO] = (
        DialogueUnderstandingTestSuiteResult.calculate_f1_micro(
            test_suite_results.command_metrics
        )
    )
    test_suite_results.f1_score[KEY_COMMANDS_F1_WEIGHTED] = (
        DialogueUnderstandingTestSuiteResult.calculate_f1_weighted(
            test_suite_results.command_metrics
        )
    )

    test_suite_results.llm_config = llm_config
    test_suite_results.latency_metrics = {
        "p50": 0.1,
        "p95": 0.2,
        "p99": 0.3,
    }
    test_suite_results.prompt_token_metrics = {
        "p50": 1000,
        "p95": 1100,
        "p99": 1200,
    }
    test_suite_results.completion_token_metrics = {
        "p50": 4,
        "p95": 5,
        "p99": 6,
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

    assert captured_span.attributes["start_tp"] == 8
    assert captured_span.attributes["start_fp"] == 2
    assert captured_span.attributes["start_fn"] == 1
    assert captured_span.attributes["start_total_count"] == 9
    assert captured_span.attributes["llm_config_0_model"] == "gpt-4"
    assert captured_span.attributes["llm_config_0_provider"] == "openai"
    assert captured_span.attributes["llm_config_0_temperature"] == 0.0
    assert captured_span.attributes["llm_config_0_timeout"] == 7
    assert captured_span.attributes["llm_config_0_top_p"] == 0.0

    assert captured_span.attributes["latency_p50"] == 0.1
    assert captured_span.attributes["latency_p95"] == 0.2
    assert captured_span.attributes["latency_p99"] == 0.3

    assert captured_span.attributes["prompt_token_p50"] == 1000
    assert captured_span.attributes["prompt_token_p95"] == 1100
    assert captured_span.attributes["prompt_token_p99"] == 1200

    assert captured_span.attributes["completion_token_p50"] == 4
    assert captured_span.attributes["completion_token_p95"] == 5
    assert captured_span.attributes["completion_token_p99"] == 6

    assert captured_span.attributes["commands_f1_macro"] == pytest.approx(
        0.8421, rel=1e-4
    )
    assert captured_span.attributes["commands_f1_micro"] == pytest.approx(
        0.8421, rel=1e-4
    )
    assert captured_span.attributes["commands_f1_weighted_average"] == pytest.approx(
        0.8421, rel=1e-4
    )
