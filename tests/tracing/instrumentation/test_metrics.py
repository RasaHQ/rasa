import json
import time
from typing import Any, Dict, Generator, List

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from rasa.tracing.constants import (
    AGENT_NAME_ATTRIBUTE_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    DURATION_UNIT_NAME,
    ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME,
    EXECUTION_CONTEXT_ATTRIBUTE_NAME,
    LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
    LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME,
    PROTOCOL_TYPE_ATTRIBUTE_NAME,
    RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME,
    RASA_CLIENT_REQUEST_DURATION_METRIC_NAME,
    REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
)
from rasa.tracing.instrumentation.metrics import (
    _mcp_agent_llm_call_metric_attributes,
    record_callable_duration_metrics,
    record_llm_based_component_cpu_usage,
    record_llm_based_component_memory_usage,
    record_llm_based_component_prompt_token,
    record_request_size_in_bytes,
)
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from rasa.utils.endpoints import EndpointConfig
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


@pytest.fixture
def endpoint_config() -> EndpointConfig:
    """Fixture providing an EndpointConfig instance for testing."""
    return EndpointConfig(url="http://localhost:5055/webhook")


def _get_latest_url_attribute_from_metrics(
    metric_reader: InMemoryMetricReader,
) -> str:
    """Helper function to extract the latest url attribute from metrics.

    Args:
        metric_reader: The InMemoryMetricReader containing the metrics data.

    Returns:
        The url attribute value from the most recent data point.

    Raises:
        AssertionError: If the metric or url attribute is not found.
    """
    metrics_data = metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]

    metrics = scope_metrics.get("metrics")
    duration_metric = find_metric_by_name(
        metrics, RASA_CLIENT_REQUEST_DURATION_METRIC_NAME
    )

    assert (
        duration_metric is not None
    ), "RASA_CLIENT_REQUEST_DURATION_METRIC_NAME not found"
    assert duration_metric.get("unit") == DURATION_UNIT_NAME

    data_points = duration_metric.get("data", {}).get("data_points")
    assert len(data_points) > 0, "No data points found"

    # Get the most recent data point
    latest_data_point = data_points[-1]
    attributes = latest_data_point.get("attributes", {})
    url = attributes.get("url")

    return url


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


# this test case must be first among the three test cases below
# to avoid interference from the other test cases
def test_record_callable_duration_metrics_endpoint_config_url_missing_key(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
    endpoint_config: EndpointConfig,
) -> None:
    """Test that when url key is missing from kwargs, it defaults to string "None".

    This test verifies that when the url key is not provided in kwargs,
    it defaults to the string "None" instead of being None.
    """
    # arrange
    start_time = time.perf_counter_ns()
    end_time = time.perf_counter_ns()

    # act
    record_callable_duration_metrics(endpoint_config, start_time, end_time)

    # assert
    url = _get_latest_url_attribute_from_metrics(in_memory_metric_reader)

    assert url is not None, "url attribute should never be None in metrics"
    assert isinstance(url, str), "url attribute should be a string"
    assert url == "None", f"Expected url to be 'None' when key is missing, got: {url}"


def test_record_callable_duration_metrics_endpoint_config_url_none_converted_to_string(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
    endpoint_config: EndpointConfig,
) -> None:
    """Test that url=None in kwargs is converted to string "None" in metrics.

    This test verifies the fix for a bug where the url attribute could be
    recorded as None value. When url=None is passed, it should be converted
    to the string "None".
    """
    # arrange
    start_time = time.perf_counter_ns()
    end_time = time.perf_counter_ns()

    # act
    record_callable_duration_metrics(endpoint_config, start_time, end_time, url=None)

    # assert
    url = _get_latest_url_attribute_from_metrics(in_memory_metric_reader)

    # The key assertion: url should never be None
    assert url is not None, "url attribute should never be None in metrics"
    # If url was None, it should be converted to the string "None"
    assert isinstance(url, str), "url attribute should be a string"
    assert url == "None", f"Expected url to be 'None' when None is passed, got: {url}"


def test_record_callable_duration_metrics_endpoint_config_url_valid_string(
    test_meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
    endpoint_config: EndpointConfig,
) -> None:
    """Test that a valid url string is recorded correctly in metrics."""
    # arrange
    start_time = time.perf_counter_ns()
    end_time = time.perf_counter_ns()
    test_url = "http://example.com"

    # act
    record_callable_duration_metrics(
        endpoint_config, start_time, end_time, url=test_url
    )

    # assert
    url = _get_latest_url_attribute_from_metrics(in_memory_metric_reader)

    assert url is not None, "url attribute should never be None in metrics"
    assert isinstance(url, str), "url attribute should be a string"
    assert url == test_url, f"Expected url to be '{test_url}', got: {url}"


def test_mcp_agent_llm_call_metric_attributes_includes_call_dimensions() -> None:
    """MCP LLM histograms only attach agent, execution context, and protocol."""
    attrs: Dict[str, Any] = {
        AGENT_NAME_ATTRIBUTE_NAME: "booking_agent",
        EXECUTION_CONTEXT_ATTRIBUTE_NAME: "agent",
        PROTOCOL_TYPE_ATTRIBUTE_NAME: "ProtocolType.MCP_TASK",
        "llm_model": "gpt-4o",
        "llm_type": "openai",
    }
    assert _mcp_agent_llm_call_metric_attributes(attrs) == {
        AGENT_NAME_ATTRIBUTE_NAME: "booking_agent",
        EXECUTION_CONTEXT_ATTRIBUTE_NAME: "agent",
        PROTOCOL_TYPE_ATTRIBUTE_NAME: "ProtocolType.MCP_TASK",
    }


def test_mcp_agent_llm_call_metric_attributes_omits_none() -> None:
    """Unset agent name is omitted from MCP LLM metric attributes."""
    attrs: Dict[str, Any] = {
        AGENT_NAME_ATTRIBUTE_NAME: None,
        EXECUTION_CONTEXT_ATTRIBUTE_NAME: "agent",
        PROTOCOL_TYPE_ATTRIBUTE_NAME: "ProtocolType.MCP_OPEN",
    }
    assert _mcp_agent_llm_call_metric_attributes(attrs) == {
        EXECUTION_CONTEXT_ATTRIBUTE_NAME: "agent",
        PROTOCOL_TYPE_ATTRIBUTE_NAME: "ProtocolType.MCP_OPEN",
    }
