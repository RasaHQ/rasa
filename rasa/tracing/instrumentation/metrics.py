from typing import Any, Dict

import psutil

from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
from rasa.core.policies.intentless_policy import IntentlessPolicy
from rasa.dialogue_understanding.generator import (
    CompactLLMCommandGenerator,
    LLMCommandGenerator,
    MultiStepLLMCommandGenerator,
    SearchReadyLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.tracing.constants import (
    COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME,
    INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
    LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    MCP_AGENT_LLM_CPU_USAGE_METRIC_NAME,
    MCP_AGENT_LLM_MEMORY_USAGE_METRIC_NAME,
    MCP_AGENT_LLM_PROMPT_TOKEN_USAGE_METRIC_NAME,
    MCP_AGENT_LLM_RESPONSE_DURATION_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME,
    RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME,
    RASA_CLIENT_REQUEST_DURATION_METRIC_NAME,
    REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
)
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from rasa.utils.endpoints import EndpointConfig


def record_llm_based_component_cpu_usage(
    metric_instrument_provider: MetricInstrumentProvider,
    metric_name: str,
) -> None:
    """Record CPU usage as a percentage.

    The recording is done by the opentelemetry.metrics.Histogram instrument.
    This instrument is registered to the MetricInstrumentProvider internal singleton.

    :param metric_instrument_provider: The MetricInstrumentProvider instance
    :param metric_name: The name of the metric instrument
    :return: None
    """
    metric_instrument = metric_instrument_provider.get_instrument(metric_name)
    if not metric_instrument:
        return None

    cpu_usage = psutil.cpu_percent()
    metric_instrument.record(amount=cpu_usage)

    return None


def record_llm_based_component_memory_usage(
    metric_instrument_provider: MetricInstrumentProvider,
    metric_name: str,
) -> None:
    """Record memory usage as a percentage.

    The recording is done by the opentelemetry.metrics.Histogram instrument.
    This instrument is registered to the MetricInstrumentProvider internal singleton.

    :param metric_instrument_provider: The MetricInstrumentProvider instance
    :param metric_name: The name of the metric instrument
    :return: None
    """
    metric_instrument = metric_instrument_provider.get_instrument(metric_name)
    if not metric_instrument:
        return None

    memory_usage = psutil.virtual_memory().percent
    metric_instrument.record(amount=memory_usage)

    return None


def record_llm_based_component_prompt_token(
    metric_instrument_provider: MetricInstrumentProvider,
    attributes: Dict[str, Any],
    metric_name: str,
) -> None:
    """
    Record prompt token length.

    The recording is done by the opentelemetry.metrics.Histogram instrument.
    This instrument is registered to the MetricInstrumentProvider internal singleton.

    :param metric_instrument_provider: The MetricInstrumentProvider instance
    :param attributes: Extracted tracing attributes
    :param metric_name: The name of the metric instrument
    :return: None
    """
    if PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME not in attributes:
        return None

    metric_instrument = metric_instrument_provider.get_instrument(metric_name)
    if not metric_instrument:
        return None

    prompt_tokens_len = attributes[PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME]

    try:
        prompt_tokens_len = int(prompt_tokens_len)
    except ValueError:
        return None

    metric_instrument.record(
        amount=prompt_tokens_len,
    )

    return None


def record_llm_command_generator_metrics(attributes: Dict[str, Any]) -> None:
    """
    Record measurements for LLMCommandGenerator specific metrics.

    The recording is done by the opentelemetry.metrics.Histogram instruments.
    These instruments are registered to the MetricInstrumentProvider internal singleton.

    :param attributes: Extracted tracing attributes
    :return: None
    """
    instrument_provider = MetricInstrumentProvider()

    if not instrument_provider.instruments:
        return None

    record_llm_based_component_cpu_usage(
        instrument_provider, LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_component_memory_usage(
        instrument_provider, LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_component_prompt_token(
        instrument_provider,
        attributes,
        LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )
    return None


def record_single_step_llm_command_generator_metrics(
    attributes: Dict[str, Any],
) -> None:
    """
    Record measurements for SingleStepLLMCommandGenerator specific metrics.

    The recording is done by the opentelemetry.metrics.Histogram instruments.
    These instruments are registered to the MetricInstrumentProvider internal singleton.

    :param attributes: Extracted tracing attributes
    :return: None
    """
    instrument_provider = MetricInstrumentProvider()

    if not instrument_provider.instruments:
        return None

    record_llm_based_component_cpu_usage(
        instrument_provider, SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_component_memory_usage(
        instrument_provider, SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_component_prompt_token(
        instrument_provider,
        attributes,
        SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )
    return None


def record_compact_llm_command_generator_metrics(
    attributes: Dict[str, Any],
) -> None:
    """
    Record measurements for CompactLLMCommandGenerator specific metrics.

    The recording is done by the opentelemetry.metrics.Histogram instruments.
    These instruments are registered to the MetricInstrumentProvider internal singleton.

    :param attributes: Extracted tracing attributes
    :return: None
    """
    instrument_provider = MetricInstrumentProvider()

    if not instrument_provider.instruments:
        return None

    record_llm_based_component_cpu_usage(
        instrument_provider, COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_component_memory_usage(
        instrument_provider, COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_component_prompt_token(
        instrument_provider,
        attributes,
        COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )
    return None


def record_search_ready_llm_command_generator_metrics(
    attributes: Dict[str, Any],
) -> None:
    """
    Record measurements for SearchReadyLLMCommandGenerator specific metrics.

    The recording is done by the opentelemetry.metrics.Histogram instruments.
    These instruments are registered to the MetricInstrumentProvider internal singleton.

    :param attributes: Extracted tracing attributes
    :return: None
    """
    instrument_provider = MetricInstrumentProvider()

    if not instrument_provider.instruments:
        return None

    record_llm_based_component_cpu_usage(
        instrument_provider, SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_component_memory_usage(
        instrument_provider, SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_component_prompt_token(
        instrument_provider,
        attributes,
        SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )
    return None


def record_multi_step_llm_command_generator_metrics(attributes: Dict[str, Any]) -> None:
    """
    Record measurements for MultiStepLLMCommandGenerator specific metrics.

    The recording is done by the opentelemetry.metrics.Histogram instruments.
    These instruments are registered to the MetricInstrumentProvider internal singleton.

    :param attributes: Extracted tracing attributes
    :return: None
    """
    instrument_provider = MetricInstrumentProvider()

    if not instrument_provider.instruments:
        return None

    record_llm_based_component_cpu_usage(
        instrument_provider, MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_component_memory_usage(
        instrument_provider, MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_component_prompt_token(
        instrument_provider,
        attributes,
        MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )
    return None


def record_enterprise_search_policy_metrics(attributes: Dict[str, Any]) -> None:
    """
    Record measurements for EnterpriseSearchPolicy specific metrics.

    The recording is done by the opentelemetry.metrics.Histogram instruments.
    These instruments are registered to the MetricInstrumentProvider internal singleton.

    :param attributes: Extracted tracing attributes
    :return: None
    """
    instrument_provider = MetricInstrumentProvider()

    if not instrument_provider.instruments:
        return None

    record_llm_based_component_cpu_usage(
        instrument_provider, ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_component_memory_usage(
        instrument_provider, ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_component_prompt_token(
        instrument_provider,
        attributes,
        ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )
    return None


def record_callable_duration_metrics(
    self: Any, start_time: int, end_time: int, **kwargs: Any
) -> None:
    """
    Record duration of instrumented method calls invoked for the following components:
    - LLMCommandGenerator
    - SingleStepLLMCommandGenerator
    - CompactLLMCommandGenerator
    - MultiStepLLMCommandGenerator
    - EnterpriseSearchPolicy
    - IntentlessPolicy
    - ContextualResponseRephraser
    - EndpointConfig

    :param self: The instance on which the instrumented method is called on.
    :param start_time: Start time measured by time.perf_counter_ns()
    before the method is called.
    :param end_time: End time measured by time.perf_counter_ns()
    after the method is called.
    :return: None
    """
    instrument_provider = MetricInstrumentProvider()
    if not instrument_provider.instruments:
        return None

    metric_instrument = None
    attributes = {}

    if type(self) == LLMCommandGenerator:
        metric_instrument = instrument_provider.get_instrument(
            LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME
        )

    if type(self) == SingleStepLLMCommandGenerator:
        metric_instrument = instrument_provider.get_instrument(
            SINGLE_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME
        )

    if type(self) == CompactLLMCommandGenerator:
        metric_instrument = instrument_provider.get_instrument(
            COMPACT_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME
        )

    if type(self) == SearchReadyLLMCommandGenerator:
        metric_instrument = instrument_provider.get_instrument(
            SEARCH_READY_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME
        )

    if type(self) == MultiStepLLMCommandGenerator:
        metric_instrument = instrument_provider.get_instrument(
            MULTI_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME
        )

    if isinstance(self, EnterpriseSearchPolicy):
        metric_instrument = instrument_provider.get_instrument(
            ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME
        )

    if isinstance(self, IntentlessPolicy):
        metric_instrument = instrument_provider.get_instrument(
            INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME
        )

    if isinstance(self, ContextualResponseRephraser):
        metric_instrument = instrument_provider.get_instrument(
            CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME
        )

    if isinstance(self, EndpointConfig):
        metric_instrument = instrument_provider.get_instrument(
            RASA_CLIENT_REQUEST_DURATION_METRIC_NAME
        )
        attributes = {"url": kwargs.get("url")}

    if isinstance(self, MCPBaseAgent):
        metric_instrument = instrument_provider.get_instrument(
            MCP_AGENT_LLM_RESPONSE_DURATION_METRIC_NAME
        )

    if not metric_instrument:
        return None

    timedelta_ms = (end_time - start_time) / 1000000
    metric_instrument.record(amount=timedelta_ms, attributes=attributes)


def record_request_size_in_bytes(attributes: Dict[str, Any]) -> None:
    """
    Record endpoint request size in bytes for EndpointConfig class.

    The recording is done by the opentelemetry.metrics.Histogram instruments.
    These instruments are registered to the MetricInstrumentProvider internal singleton.

    :param attributes: Extracted tracing attributes
    :return: None
    """

    metric_instrument_provider = MetricInstrumentProvider()

    if not metric_instrument_provider.instruments:
        return None

    metric_instrument = metric_instrument_provider.get_instrument(
        RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME
    )
    if not metric_instrument:
        return None

    request_body_size = attributes.pop(REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME, 0)
    metric_instrument.record(amount=request_body_size, attributes=attributes)


def record_mcp_agent_llm_metrics(attributes: Dict[str, Any]) -> None:
    """Record MCP agent LLM metrics."""
    instrument_provider = MetricInstrumentProvider()

    if not instrument_provider.instruments:
        return None

    # Use MCP agent specific metric names
    record_llm_based_component_cpu_usage(
        instrument_provider, MCP_AGENT_LLM_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_component_memory_usage(
        instrument_provider, MCP_AGENT_LLM_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_component_prompt_token(
        instrument_provider,
        attributes,
        MCP_AGENT_LLM_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )
    return None
