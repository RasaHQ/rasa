import inspect
import json
import time
from typing import Generator, Any, Optional, Dict, Callable, List
from unittest.mock import Mock

import opentelemetry
import pytest
import requests
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from pytest import MonkeyPatch
from opentelemetry.sdk.trace import TracerProvider

from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
from rasa.core.policies.intentless_policy import IntentlessPolicy
from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy

from rasa.dialogue_understanding.generator import (
    LLMCommandGenerator,
    MultiStepLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import OPENAI_API_KEY_ENV_VAR
from rasa.shared.core.domain import Domain
from rasa.tracing.constants import (
    LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    DURATION_UNIT_NAME,
    ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
    INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
    CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME,
    RASA_CLIENT_REQUEST_DURATION_METRIC_NAME,
    LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
)
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from rasa.utils.endpoints import EndpointConfig
from tests.integration_tests.conftest import send_message_to_rasa_server

ACTION_SERVER_TEST_URL = "http://localhost:5055/webhook"
RASA_SERVER_TEST_URL = "http://localhost:5005"
OTLP_METRICS_TEST_URL = "http://localhost:8889/metrics"
PROMETHEUS_METRICS_QUERY_TEST_URL = "http://localhost:9090/api/v1/metadata"


@pytest.fixture(autouse=True)
def set_mock_openai_api_key(monkeypatch: MonkeyPatch):
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_metrics")


@pytest.fixture(scope="module")
def tracer_provider() -> TracerProvider:
    return TracerProvider()


@pytest.fixture(scope="module")
def metric_reader() -> Generator[InMemoryMetricReader, None, None]:
    metric_reader = InMemoryMetricReader()
    yield metric_reader
    metric_reader.force_flush()


@pytest.fixture(scope="module")
def meter_provider(metric_reader: MetricReader) -> Generator[MeterProvider, None, None]:
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    opentelemetry.metrics.set_meter_provider(meter_provider)
    instrument_provider = MetricInstrumentProvider()
    instrument_provider.register_instruments()

    yield meter_provider

    meter_provider.shutdown()


def setup_test_llm_command_generator(
    monkeypatch: MonkeyPatch,
    **kwargs: Any,
) -> LLMCommandGenerator:
    async def mock_llm_command_generate(self: Any, prompt: str) -> Optional[str]:
        return ""

    monkeypatch.setattr(
        LLMCommandGenerator,
        "invoke_llm",
        mock_llm_command_generate,
    )

    default_model_storage = kwargs.get("default_model_storage")

    return LLMCommandGenerator(
        {}, default_model_storage, Resource("test_llm_command_generator")
    )


def setup_test_single_step_llm_command_generator(
    monkeypatch: MonkeyPatch,
    **kwargs: Any,
) -> SingleStepLLMCommandGenerator:
    async def mock_single_step_llm_command_generate(
        self: Any, prompt: str
    ) -> Optional[str]:
        return ""

    monkeypatch.setattr(
        SingleStepLLMCommandGenerator,
        "invoke_llm",
        mock_single_step_llm_command_generate,
    )

    default_model_storage = kwargs.get("default_model_storage")

    return SingleStepLLMCommandGenerator(
        {}, default_model_storage, Resource("test_single_step_llm_command_generator")
    )


def setup_test_multi_step_llm_command_generator(
    monkeypatch: MonkeyPatch,
    **kwargs: Any,
) -> MultiStepLLMCommandGenerator:
    async def mock_multi_step_llm_command_generate(
        self: Any, prompt: str
    ) -> Optional[str]:
        return ""

    monkeypatch.setattr(
        MultiStepLLMCommandGenerator,
        "invoke_llm",
        mock_multi_step_llm_command_generate,
    )

    default_model_storage = kwargs.get("default_model_storage")

    return MultiStepLLMCommandGenerator(
        {}, default_model_storage, Resource("test_multi_step_llm_command_generator")
    )


def setup_test_enterprise_search_policy(
    monkeypatch: MonkeyPatch,
    **kwargs: Any,
) -> EnterpriseSearchPolicy:
    def mock_enterprise_search_generate(
        self: Any, llm: Any, prompt: str
    ) -> Optional[str]:
        return ""

    monkeypatch.setattr(
        EnterpriseSearchPolicy, "_generate_llm_answer", mock_enterprise_search_generate
    )

    default_model_storage = kwargs.get("default_model_storage")
    default_execution_context = kwargs.get("default_execution_context")

    return EnterpriseSearchPolicy(
        {},
        default_model_storage,
        Resource("test_enterprise_search"),
        default_execution_context,
    )


def setup_test_intentless_policy(
    monkeypatch: MonkeyPatch,
    **kwargs: Any,
) -> IntentlessPolicy:
    def mock_intentless_generate(self: Any, llm: Any, prompt: str) -> Optional[str]:
        return ""

    def mock_constructor(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.trace_prompt_tokens = False

    monkeypatch.setattr(
        IntentlessPolicy, "_generate_llm_answer", mock_intentless_generate
    )
    monkeypatch.setattr(IntentlessPolicy, "__init__", mock_constructor)

    default_model_storage = kwargs.get("default_model_storage")
    default_execution_context = kwargs.get("default_execution_context")

    return IntentlessPolicy(
        {},
        default_model_storage,
        Resource("test_intentless"),
        default_execution_context,
    )


def setup_test_contextual_nlg(
    monkeypatch: MonkeyPatch,
    **kwargs: Any,
) -> ContextualResponseRephraser:
    async def mock_contextual_nlg_generate(self: Any, prompt: str) -> Optional[str]:
        return ""

    monkeypatch.setattr(
        ContextualResponseRephraser,
        "_generate_llm_response",
        mock_contextual_nlg_generate,
    )

    return ContextualResponseRephraser(EndpointConfig(), Domain.empty())


def setup_test_endpoint_config(
    monkeypatch: MonkeyPatch,
    **kwargs: Any,
) -> EndpointConfig:
    async def mock_request(
        self,
        method: str = "post",
        subpath: Optional[str] = None,
        content_type: Optional[str] = "application/json",
        compress: bool = False,
        **kwargs: Any,
    ) -> Optional[Any]:
        return None

    monkeypatch.setattr(EndpointConfig, "request", mock_request)

    return EndpointConfig(url=ACTION_SERVER_TEST_URL)


@pytest.mark.parametrize(
    "instrumentation_arg, component_class, setup_component, setup_args, method_name, input_args, metric_index, metric_name, metric_description",  # noqa: E501
    [
        (
            "llm_command_generator_class",
            LLMCommandGenerator,
            setup_test_llm_command_generator,
            ["default_model_storage"],
            "invoke_llm",
            ["prompt"],
            0,
            LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
            "The duration of LLMCommandGenerator's LLM call",
        ),
        (
            "policy_subclasses",
            EnterpriseSearchPolicy,
            setup_test_enterprise_search_policy,
            ["default_model_storage", "default_execution_context"],
            "_generate_llm_answer",
            ["llm", "prompt"],
            # the first 3 metrics in the list belong to LLMCommandGenerator
            3,
            ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
            "The duration of EnterpriseSearchPolicy's LLM call",
        ),
        (
            "policy_subclasses",
            IntentlessPolicy,
            setup_test_intentless_policy,
            ["default_model_storage", "default_execution_context"],
            "_generate_llm_answer",
            ["llm", "prompt"],
            4,
            INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
            "The duration of IntentlessPolicy's LLM call",
        ),
        (
            "contextual_response_rephraser_class",
            ContextualResponseRephraser,
            setup_test_contextual_nlg,
            [],
            "_generate_llm_response",
            ["prompt"],
            5,
            CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME,
            "The duration of ContextualResponseRephraser's LLM call",
        ),
        (
            "endpoint_config_class",
            EndpointConfig,
            setup_test_endpoint_config,
            [],
            "request",
            ["json"],
            6,
            RASA_CLIENT_REQUEST_DURATION_METRIC_NAME,
            "The duration of the rasa client request",
        ),
        (
            "multi_step_llm_command_generator_class",
            MultiStepLLMCommandGenerator,
            setup_test_multi_step_llm_command_generator,
            ["default_model_storage"],
            "invoke_llm",
            ["prompt"],
            8,
            MULTI_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
            "The duration of MultiStepLLMCommandGenerator's LLM call",
        ),
        (
            "single_step_llm_command_generator_class",
            SingleStepLLMCommandGenerator,
            setup_test_single_step_llm_command_generator,
            ["default_model_storage"],
            "invoke_llm",
            ["prompt"],
            11,
            SINGLE_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
            "The duration of SingleStepLLMCommandGenerator's LLM call",
        ),
    ],
)
async def test_record_callable_duration_metrics(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    metric_reader: InMemoryMetricReader,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: MonkeyPatch,
    instrumentation_arg: str,
    component_class: Any,
    setup_component: Callable,
    setup_args: List[str],
    method_name: str,
    input_args: List[str],
    metric_index: int,
    metric_name: str,
    metric_description: str,
) -> None:
    """Test record_callable_duration_metrics for all supported components."""
    # arrange
    setup_kwargs = {}
    for arg in setup_args:
        if arg == "default_model_storage":
            setup_kwargs[arg] = default_model_storage
        if arg == "default_execution_context":
            setup_kwargs[arg] = default_execution_context

    test_component = setup_component(monkeypatch, **setup_kwargs)

    instrumentation_kwargs = (
        {instrumentation_arg: [component_class]}
        if instrumentation_arg == "policy_subclasses"
        else {instrumentation_arg: component_class}
    )

    instrumentation.instrument(
        tracer_provider,
        **instrumentation_kwargs,
    )

    # act
    kwargs = {}
    for arg in input_args:
        if arg == "llm":
            kwargs[arg] = Mock()
        if arg == "prompt":
            kwargs[arg] = "some text"
        if arg == "json":
            kwargs[arg] = {"test": "value"}

    method = getattr(test_component, method_name)

    if inspect.iscoroutinefunction(method):
        await method(**kwargs)
    else:
        method(**kwargs)

    # assert
    metrics_data = metric_reader.get_metrics_data()
    metrics_data = json.loads(metrics_data.to_json())

    resource_metrics = metrics_data.get("resource_metrics")[0]
    scope_metrics = resource_metrics.get("scope_metrics")[0]
    metrics = scope_metrics.get("metrics")[metric_index]

    assert metrics.get("name") == metric_name
    assert metrics.get("description") == metric_description
    assert metrics.get("unit") == DURATION_UNIT_NAME

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert data_points.get("min") > 0
    assert data_points.get("max") > 0

    url = data_points.get("attributes", {}).get("url")
    if url is not None:
        assert url == ACTION_SERVER_TEST_URL


@pytest.fixture(scope="module")
def send_user_message() -> str:
    sender_id, _ = send_message_to_rasa_server(RASA_SERVER_TEST_URL, "list my contacts")
    return sender_id


def test_metrics_get_sent_to_otlp_collector(
    send_user_message: str,
) -> None:
    # make sure the OTLP collector has collected the metrics
    while True:
        response = requests.get(OTLP_METRICS_TEST_URL)
        assert response.status_code == 200

        if len(response.text) > 0:
            break

    assert LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME in response.text
    assert LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME in response.text
    assert LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME in response.text
    assert ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME in response.text
    assert INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME in response.text
    assert (
        CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME in response.text
    )
    assert LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME in response.text


def test_metrics_get_sent_to_prometheus(
    send_user_message: str,
) -> None:
    # make sure that Prometheus has collected the metrics
    time.sleep(30)
    while True:
        prometheus_response = requests.get(PROMETHEUS_METRICS_QUERY_TEST_URL)
        assert prometheus_response.status_code == 200
        data = prometheus_response.json().get("data", {})
        assert len(data) > 0

        data_keys = list(data.keys())

        # filter out the default otelcol metrics
        filtered_metrics = list(filter(lambda x: "otelcol" not in x, data_keys))

        # filtered metrics contain another default metric `target_info`,
        # so we need to check if there are more than 1
        if len(filtered_metrics) > 1:
            break

    data_values = list(data.values())

    # metric names exported to Prometheus always contain the converted unit
    # as a suffix, so we need to check for the metric names with the unit
    # source: https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/compatibility/prometheus_and_openmetrics.md#metric-metadata-1  # noqa: E501
    assert "contextual_nlg_llm_response_duration_milliseconds" in data_keys
    assert [
        {
            "type": "histogram",
            "unit": "",
            "help": "The duration of ContextualResponseRephraser's LLM call",
        }
    ] in data_values

    assert "enterprise_search_policy_llm_response_duration_milliseconds" in data_keys
    assert [
        {
            "type": "histogram",
            "unit": "",
            "help": "The duration of EnterpriseSearchPolicy's LLM call",
        }
    ] in data_values

    assert "intentless_policy_llm_response_duration_milliseconds" in data_keys
    assert [
        {
            "type": "histogram",
            "unit": "",
            "help": "The duration of IntentlessPolicy's LLM call",
        }
    ] in data_values

    assert "llm_command_generator_cpu_usage_percentage" in data_keys
    assert [
        {
            "type": "histogram",
            "unit": "",
            "help": "CPU percentage for LLMCommandGenerator",
        }
    ] in data_values

    assert "llm_command_generator_llm_response_duration_milliseconds" in data_keys
    assert [
        {
            "type": "histogram",
            "unit": "",
            "help": "The duration of LLMCommandGenerator's LLM call",
        }
    ] in data_values

    assert "llm_command_generator_memory_usage_percentage" in data_keys
    assert [
        {
            "type": "histogram",
            "unit": "",
            "help": "RAM memory usage for LLMCommandGenerator",
        }
    ] in data_values

    assert "llm_command_generator_prompt_token_usage" in data_keys
    assert [
        {
            "type": "histogram",
            "unit": "",
            "help": "LLMCommandGenerator prompt token length",
        }
    ] in data_values

    assert "rasa_client_request_duration_milliseconds" in data_keys
    assert [
        {
            "type": "histogram",
            "unit": "",
            "help": "The duration of the rasa client request",
        }
    ] in data_values

    assert "rasa_client_request_body_size_byte" in data_keys
    assert [
        {
            "type": "histogram",
            "unit": "",
            "help": "The rasa client request's body size",
        }
    ] in data_values
