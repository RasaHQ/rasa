import json
from typing import Any, Dict, Generator, List

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from rasa.tracing.constants import (
    COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME,
    LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
    LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME,
    RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME,
    REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
)
from rasa.tracing.instrumentation.metrics import (
    record_llm_based_component_cpu_usage,
    record_llm_based_component_memory_usage,
    record_llm_based_component_prompt_token,
    record_request_size_in_bytes,
)
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from tests.tracing.conftest import set_up_test_meter_provider


def find_metric_by_name(
    metrics_list: List[Dict[str, Any]], metric_name: str
) -> Dict[str, Any]:
    """Helper function to find a metric by name in the metrics list."""
    for metric in metrics_list:
        if metric.get("name") == metric_name:
            return metric
    raise AssertionError(f"Metric {metric_name} not found in metrics list")


@pytest.fixture(scope="module")
def in_memory_metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture(scope="module")
def test_meter_provider(
    in_memory_metric_reader: InMemoryMetricReader,
) -> Generator[MeterProvider, None, None]:
    meter_provider = next(set_up_test_meter_provider(in_memory_metric_reader))

    instrument_provider = MetricInstrumentProvider()
    instrument_provider.register_instruments()

    yield meter_provider
    meter_provider.shutdown()


def test_record_llm_command_generator_cpu_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_cpu_usage(
        MetricInstrumentProvider(), LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    assert metrics.get("description") == "CPU percentage for LLMCommandGenerator"
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_llm_command_generator_memory_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_memory_usage(
        MetricInstrumentProvider(), LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the LLM command generator memory usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    assert metrics.get("description") == "RAM memory usage for LLMCommandGenerator"
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_llm_command_generator_prompt_token_exists(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # arrange
    prompt_token_len = 500

    # act
    record_llm_based_component_prompt_token(
        MetricInstrumentProvider(),
        attributes={PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME: prompt_token_len},
        metric_name=LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the LLM command generator prompt token usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME
    )
    assert metrics.get("description") == "LLMCommandGenerator prompt token length"
    assert metrics.get("unit") == "1"

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert (
        data_points.get("sum")
        == data_points.get("min")
        == data_points.get("max")
        == prompt_token_len
    )


def test_record_multi_step_llm_command_generator_cpu_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_cpu_usage(
        MetricInstrumentProvider(),
        MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the multi-step LLM command generator CPU usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description") == "CPU percentage for MultiStepLLMCommandGenerator"
    )
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_multi_step_llm_command_generator_memory_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_memory_usage(
        MetricInstrumentProvider(),
        MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the multi-step LLM command generator memory usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description")
        == "RAM memory usage for MultiStepLLMCommandGenerator"
    )
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_multi_step_llm_command_generator_prompt_token_exists(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # arrange
    prompt_token_len = 500

    # act
    record_llm_based_component_prompt_token(
        MetricInstrumentProvider(),
        attributes={PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME: prompt_token_len},
        metric_name=MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the multi-step LLM command generator prompt token usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description") == "MultiStepLLMCommandGenerator prompt token length"
    )
    assert metrics.get("unit") == "1"

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert (
        data_points.get("sum")
        == data_points.get("min")
        == data_points.get("max")
        == prompt_token_len
    )


def test_record_single_step_llm_command_generator_cpu_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_cpu_usage(
        MetricInstrumentProvider(),
        SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the single-step LLM command generator CPU usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description") == "CPU percentage for SingleStepLLMCommandGenerator"
    )
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_single_step_llm_command_generator_memory_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_memory_usage(
        MetricInstrumentProvider(),
        SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the single-step LLM command generator memory usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description")
        == "RAM memory usage for SingleStepLLMCommandGenerator"
    )
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_single_step_llm_command_generator_prompt_token_exists(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # arrange
    prompt_token_len = 500

    # act
    record_llm_based_component_prompt_token(
        MetricInstrumentProvider(),
        attributes={PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME: prompt_token_len},
        metric_name=SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the single-step LLM command generator prompt token usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description")
        == "SingleStepLLMCommandGenerator prompt token length"
    )
    assert metrics.get("unit") == "1"

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert (
        data_points.get("sum")
        == data_points.get("min")
        == data_points.get("max")
        == prompt_token_len
    )


def test_record_compact_llm_command_generator_cpu_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_cpu_usage(
        MetricInstrumentProvider(),
        COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the compact_llm_command_generator_cpu_usage metric in the list
    metrics_list = scope_metrics.get("metrics")
    metrics = None
    for metric in metrics_list:
        if metric.get("name") == COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME:
            metrics = metric
            break

    assert metrics is not None, (
        f"Metric {COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME} "
        "not found in metrics list"
    )
    assert metrics.get("name") == COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    assert metrics.get("description") == "CPU percentage for CompactLLMCommandGenerator"
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_compact_llm_command_generator_memory_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_memory_usage(
        MetricInstrumentProvider(),
        COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the compact LLM command generator memory usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description") == "RAM memory usage for CompactLLMCommandGenerator"
    )
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_compact_llm_command_generator_prompt_token_exists(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # arrange
    prompt_token_len = 500

    # act
    record_llm_based_component_prompt_token(
        MetricInstrumentProvider(),
        attributes={PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME: prompt_token_len},
        metric_name=COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the compact LLM command generator prompt token usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description") == "CompactLLMCommandGenerator prompt token length"
    )
    assert metrics.get("unit") == "1"

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert (
        data_points.get("sum")
        == data_points.get("min")
        == data_points.get("max")
        == prompt_token_len
    )


def test_record_search_ready_llm_command_generator_cpu_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_cpu_usage(
        MetricInstrumentProvider(),
        SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the search ready LLM command generator CPU usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("name") == SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description")
        == "CPU percentage for SearchReadyLLMCommandGenerator"
    )
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_search_ready_llm_command_generator_memory_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_memory_usage(
        MetricInstrumentProvider(),
        SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the search ready LLM command generator memory usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description")
        == "RAM memory usage for SearchReadyLLMCommandGenerator"
    )
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_search_ready_llm_command_generator_prompt_token_exists(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # arrange
    prompt_token_len = 500

    # act
    record_llm_based_component_prompt_token(
        MetricInstrumentProvider(),
        attributes={PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME: prompt_token_len},
        metric_name=SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the search ready LLM command generator prompt token usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME
    )
    assert (
        metrics.get("description")
        == "SearchReadyLLMCommandGenerator prompt token length"
    )
    assert metrics.get("unit") == "1"

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert (
        data_points.get("sum")
        == data_points.get("min")
        == data_points.get("max")
        == prompt_token_len
    )


def test_record_enterprise_search_policy_cpu_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_cpu_usage(
        MetricInstrumentProvider(),
        ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the enterprise_search_policy_cpu_usage metric in the list
    metrics_list = scope_metrics.get("metrics")
    metrics = None
    for metric in metrics_list:
        if metric.get("name") == ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME:
            metrics = metric
            break

    assert metrics is not None, (
        f"Metric {ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME} "
        "not found in metrics list"
    )
    assert metrics.get("name") == ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME
    assert metrics.get("description") == "CPU percentage for EnterpriseSearchPolicy"
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_enterprise_search_policy_memory_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # act
    record_llm_based_component_memory_usage(
        MetricInstrumentProvider(),
        ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the enterprise search policy memory usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME
    )
    assert metrics.get("description") == "RAM memory usage for EnterpriseSearchPolicy"
    assert metrics.get("unit") == LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert 0 < data_points.get("min") <= 100
    assert 0 <= data_points.get("max") <= 100


def test_record_enterprise_search_policy_prompt_token_usage(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # arrange
    prompt_token_len = 500

    # act
    record_llm_based_component_prompt_token(
        MetricInstrumentProvider(),
        attributes={PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME: prompt_token_len},
        metric_name=ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the enterprise search policy prompt token usage metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME
    )
    assert metrics.get("description") == "EnterpriseSearchPolicy prompt token length"
    assert metrics.get("unit") == "1"

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert (
        data_points.get("sum")
        == data_points.get("min")
        == data_points.get("max")
        == prompt_token_len
    )


def test_record_request_size_in_bytes(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> None:
    # arrange
    request_body_size = len(json.dumps({"key": "value"}).encode("utf-8"))

    # act
    record_request_size_in_bytes(
        attributes={
            "url": "http://localhost:5055/webhook",
            REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME: request_body_size,
        },
    )

    # assert
    metrics_data = in_memory_metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    # Find the Rasa client request body size metric
    metrics_list = scope_metrics.get("metrics")
    metrics = find_metric_by_name(
        metrics_list, RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME
    )
    assert metrics.get("description") == "The rasa client request's body size"
    assert metrics.get("unit") == "byte"

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert (
        data_points.get("sum")
        == data_points.get("min")
        == data_points.get("max")
        == request_body_size
    )
