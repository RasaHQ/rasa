from typing import Dict, Any

import psutil

from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
from rasa.core.policies.intentless_policy import IntentlessPolicy
from rasa.dialogue_understanding.generator import (
    LLMCommandGenerator,
    SingleStepLLMCommandGenerator,
    MultiStepLLMCommandGenerator,
)
from rasa.tracing.constants import (
    LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME,
    LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
    INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
    CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME,
    RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME,
    RASA_CLIENT_REQUEST_DURATION_METRIC_NAME,
    REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME,
)
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from rasa.utils.endpoints import EndpointConfig


def record_llm_based_command_generator_cpu_usage(
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


def record_llm_based_command_generator_memory_usage(
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


def record_llm_based_command_generator_prompt_token(
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

    record_llm_based_command_generator_cpu_usage(
        instrument_provider, LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_command_generator_memory_usage(
        instrument_provider, LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_command_generator_prompt_token(
        instrument_provider,
        attributes,
        LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )


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

    record_llm_based_command_generator_cpu_usage(
        instrument_provider, SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_command_generator_memory_usage(
        instrument_provider, SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_command_generator_prompt_token(
        instrument_provider,
        attributes,
        SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )


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

    record_llm_based_command_generator_cpu_usage(
        instrument_provider, MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME
    )
    record_llm_based_command_generator_memory_usage(
        instrument_provider, MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME
    )
    record_llm_based_command_generator_prompt_token(
        instrument_provider,
        attributes,
        MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    )


def record_callable_duration_metrics(
    self: Any, start_time: int, end_time: int, **kwargs: Any
) -> None:
    """
    Record duration of instrumented method calls invoked for the following components:
    - LLMCommandGenerator
    - SingleStepLLMCommandGenerator
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
