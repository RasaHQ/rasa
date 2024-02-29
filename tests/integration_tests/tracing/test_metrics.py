import inspect
import json
from typing import Generator, Any, Optional, Dict, Callable, List
from unittest.mock import Mock

import opentelemetry
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from pytest import MonkeyPatch
from opentelemetry.sdk.trace import TracerProvider

from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
from rasa.core.policies.intentless_policy import IntentlessPolicy
from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
from rasa.dialogue_understanding.generator import LLMCommandGenerator
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.tracing.constants import (
    LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    INTENTLESS_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    RASA_CLIENT_REQUEST_DURATION_INSTRUMENT_NAME,
)
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from rasa.utils.endpoints import EndpointConfig


TEST_URL = "http://localhost:5055/webhook"


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
    def mock_llm_command_generate(self: Any, prompt: str) -> Optional[str]:
        return ""

    monkeypatch.setattr(
        LLMCommandGenerator,
        "_generate_action_list_using_llm",
        mock_llm_command_generate,
    )

    default_model_storage = kwargs.get("default_model_storage")

    return LLMCommandGenerator(
        {}, default_model_storage, Resource("test_llm_command_generator")
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
        **kwargs,
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
    def mock_contextual_nlg_generate(self: Any, prompt: str) -> Optional[str]:
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

    return EndpointConfig(url=TEST_URL)


@pytest.mark.parametrize(
    "instrumentation_arg, component_class, setup_component, setup_args, method_name, input_args, metric_index, metric_name, metric_description",  # noqa: E501
    [
        (
            "llm_command_generator_class",
            LLMCommandGenerator,
            setup_test_llm_command_generator,
            ["default_model_storage"],
            "_generate_action_list_using_llm",
            ["prompt"],
            0,
            LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
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
            ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
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
            INTENTLESS_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
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
            CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
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
            RASA_CLIENT_REQUEST_DURATION_INSTRUMENT_NAME,
            "The duration of the rasa client request",
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
    assert metrics.get("unit") == "ms"

    data_points = metrics.get("data", {}).get("data_points")[0]
    assert data_points.get("count") == 1
    assert data_points.get("min") > 0
    assert data_points.get("max") > 0

    url = data_points.get("attributes", {}).get("url")
    if url is not None:
        assert url == TEST_URL
