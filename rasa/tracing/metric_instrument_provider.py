from typing import Any, ClassVar, Dict

from opentelemetry.metrics import get_meter_provider
from opentelemetry.sdk.metrics import Meter

from rasa.tracing.constants import (
    AGENT_EXECUTION_DURATION_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME,
    DURATION_UNIT_NAME,
    ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME,
    ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME,
    INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
    LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
    LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    MCP_AGENT_LLM_CPU_USAGE_METRIC_NAME,
    MCP_AGENT_LLM_MEMORY_USAGE_METRIC_NAME,
    MCP_AGENT_LLM_PROMPT_TOKEN_USAGE_METRIC_NAME,
    MCP_AGENT_LLM_RESPONSE_DURATION_METRIC_NAME,
    MCP_TOOL_EXECUTION_DURATION_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME,
    RASA_CLIENT_REQUEST_DURATION_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
    SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
)
from rasa.utils.singleton import Singleton


class MetricInstrumentProvider(metaclass=Singleton):
    """Singleton provider class of metric instruments."""

    instruments: ClassVar[Dict[str, Any]] = {}

    def register_instruments(self) -> None:
        """Update instruments class attribute.

        The registered instruments are subclasses of the
        opentelemetry.metrics._internal.instrument.Instrument interface.
        """
        meter = get_meter_provider().get_meter(__name__)

        instruments = {
            **self._create_llm_command_generator_instruments(meter),
            **self._create_single_step_llm_command_generator_instruments(meter),
            **self._create_compact_llm_command_generator_instruments(meter),
            **self._create_search_ready_llm_command_generator_instruments(meter),
            **self._create_multi_step_llm_command_generator_instruments(meter),
            **self._create_enterprise_search_policy_instruments(meter),
            **self._create_llm_response_duration_instruments(meter),
            **self._create_mcp_and_a2a_agent_instruments(meter),
            **self._create_mcp_agent_llm_instruments(meter),
            **self._create_client_request_instruments(meter),
        }

        self.instruments.update(instruments)

    def get_instrument(self, name: str) -> Any:
        """Get the instrument mapped to the provided name."""
        return self.instruments.get(name)

    @staticmethod
    def _create_llm_command_generator_instruments(meter: Meter) -> Dict[str, Any]:
        llm_command_generator_cpu_usage = meter.create_histogram(
            name=LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
            description="CPU percentage for LLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        llm_command_generator_memory_usage = meter.create_histogram(
            name=LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
            description="RAM memory usage for LLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        llm_command_generator_prompt_token_usage = meter.create_histogram(
            name=LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
            description="LLMCommandGenerator prompt token length",
            unit="1",
        )

        llm_response_duration_llm_command_generator = meter.create_histogram(
            name=LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
            description="The duration of LLMCommandGenerator's LLM call",
            unit=DURATION_UNIT_NAME,
        )

        return {
            LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME: llm_command_generator_cpu_usage,  # noqa: E501
            LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME: llm_command_generator_memory_usage,  # noqa: E501
            LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME: llm_command_generator_prompt_token_usage,  # noqa: E501
            LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME: llm_response_duration_llm_command_generator,  # noqa: E501
        }

    @staticmethod
    def _create_single_step_llm_command_generator_instruments(
        meter: Meter,
    ) -> Dict[str, Any]:
        single_step_llm_command_generator_cpu_usage = meter.create_histogram(
            name=SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
            description="CPU percentage for SingleStepLLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        single_step_llm_command_generator_memory_usage = meter.create_histogram(
            name=SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
            description="RAM memory usage for SingleStepLLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        single_step_llm_command_generator_prompt_token_usage = meter.create_histogram(
            name=SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
            description="SingleStepLLMCommandGenerator prompt token length",
            unit="1",
        )

        single_step_llm_response_duration_llm_command_generator = meter.create_histogram(  # noqa: E501
            name=SINGLE_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
            description="The duration of SingleStepLLMCommandGenerator's LLM call",
            unit=DURATION_UNIT_NAME,
        )

        return {
            SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME: single_step_llm_command_generator_cpu_usage,  # noqa: E501
            SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME: single_step_llm_command_generator_memory_usage,  # noqa: E501
            SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME: single_step_llm_command_generator_prompt_token_usage,  # noqa: E501
            SINGLE_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME: single_step_llm_response_duration_llm_command_generator,  # noqa: E501
        }

    @staticmethod
    def _create_compact_llm_command_generator_instruments(
        meter: Meter,
    ) -> Dict[str, Any]:
        compact_llm_command_generator_cpu_usage = meter.create_histogram(
            name=COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
            description="CPU percentage for CompactLLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        compact_llm_command_generator_memory_usage = meter.create_histogram(
            name=COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
            description="RAM memory usage for CompactLLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        compact_llm_command_generator_prompt_token_usage = meter.create_histogram(
            name=COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
            description="CompactLLMCommandGenerator prompt token length",
            unit="1",
        )

        compact_llm_response_duration_llm_command_generator = meter.create_histogram(
            name=COMPACT_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
            description="The duration of CompactLLMCommandGenerator's LLM call",
            unit=DURATION_UNIT_NAME,
        )

        return {
            COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME: compact_llm_command_generator_cpu_usage,  # noqa: E501
            COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME: compact_llm_command_generator_memory_usage,  # noqa: E501
            COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME: compact_llm_command_generator_prompt_token_usage,  # noqa: E501
            COMPACT_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME: compact_llm_response_duration_llm_command_generator,  # noqa: E501
        }

    @staticmethod
    def _create_search_ready_llm_command_generator_instruments(
        meter: Meter,
    ) -> Dict[str, Any]:
        search_ready_llm_command_generator_cpu_usage = meter.create_histogram(
            name=SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
            description="CPU percentage for SearchReadyLLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        search_ready_llm_command_generator_memory_usage = meter.create_histogram(
            name=SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
            description="RAM memory usage for SearchReadyLLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        search_ready_llm_command_generator_prompt_token_usage = meter.create_histogram(
            name=SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
            description="SearchReadyLLMCommandGenerator prompt token length",
            unit="1",
        )

        search_ready_llm_response_duration_command_generator = meter.create_histogram(
            name=SEARCH_READY_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
            description="The duration of SearchReadyLLMCommandGenerator's LLM call",
            unit=DURATION_UNIT_NAME,
        )

        return {
            SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME: search_ready_llm_command_generator_cpu_usage,  # noqa: E501
            SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME: search_ready_llm_command_generator_memory_usage,  # noqa: E501
            SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME: search_ready_llm_command_generator_prompt_token_usage,  # noqa: E501
            SEARCH_READY_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME: search_ready_llm_response_duration_command_generator,  # noqa: E501
        }

    @staticmethod
    def _create_multi_step_llm_command_generator_instruments(
        meter: Meter,
    ) -> Dict[str, Any]:
        multi_step_llm_command_generator_cpu_usage = meter.create_histogram(
            name=MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME,
            description="CPU percentage for MultiStepLLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        multi_step_llm_command_generator_memory_usage = meter.create_histogram(
            name=MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME,
            description="RAM memory usage for MultiStepLLMCommandGenerator",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        multi_step_llm_command_generator_prompt_token_usage = meter.create_histogram(
            name=MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME,
            description="MultiStepLLMCommandGenerator prompt token length",
            unit="1",
        )

        multi_step_llm_response_duration_llm_command_generator = meter.create_histogram(
            name=MULTI_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME,
            description="The duration of MultiStepLLMCommandGenerator's LLM call",
            unit=DURATION_UNIT_NAME,
        )

        return {
            MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME: multi_step_llm_command_generator_cpu_usage,  # noqa: E501
            MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME: multi_step_llm_command_generator_memory_usage,  # noqa: E501
            MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME: multi_step_llm_command_generator_prompt_token_usage,  # noqa: E501
            MULTI_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME: multi_step_llm_response_duration_llm_command_generator,  # noqa: E501
        }

    @staticmethod
    def _create_enterprise_search_policy_instruments(
        meter: Meter,
    ) -> Dict[str, Any]:
        enterprise_search_policy_cpu_usage = meter.create_histogram(
            name=ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME,
            description="CPU percentage for EnterpriseSearchPolicy",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        enterprise_search_policy_memory_usage = meter.create_histogram(
            name=ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME,
            description="RAM memory usage for EnterpriseSearchPolicy",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        enterprise_search_policy_prompt_token_usage = meter.create_histogram(
            name=ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME,
            description="EnterpriseSearchPolicy prompt token length",
            unit="1",
        )

        enterprise_search_policy_llm_response_duration = meter.create_histogram(
            name=ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
            description="The duration of EnterpriseSearchPolicy's LLM call",
            unit=DURATION_UNIT_NAME,
        )

        return {
            ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME: enterprise_search_policy_cpu_usage,  # noqa: E501
            ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME: enterprise_search_policy_memory_usage,  # noqa: E501
            ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME: enterprise_search_policy_prompt_token_usage,  # noqa: E501
            ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME: enterprise_search_policy_llm_response_duration,  # noqa: E501
        }

    @staticmethod
    def _create_llm_response_duration_instruments(meter: Meter) -> Dict[str, Any]:
        llm_response_duration_intentless = meter.create_histogram(
            name=INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME,
            description="The duration of IntentlessPolicy's LLM call",
            unit=DURATION_UNIT_NAME,
        )

        llm_response_duration_contextual_nlg = meter.create_histogram(
            name=CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME,
            description="The duration of ContextualResponseRephraser's LLM call",
            unit=DURATION_UNIT_NAME,
        )

        return {
            INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME: llm_response_duration_intentless,  # noqa: E501
            CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME: llm_response_duration_contextual_nlg,  # noqa: E501
        }

    @staticmethod
    def _create_mcp_and_a2a_agent_instruments(
        meter: Meter,
    ) -> Dict[str, Any]:
        """Create instruments for MCP tool execution and agent execution."""
        mcp_tool_execution_duration = meter.create_histogram(
            name=MCP_TOOL_EXECUTION_DURATION_METRIC_NAME,
            description="The duration of MCP tool execution",
            unit=DURATION_UNIT_NAME,
        )

        agent_execution_duration = meter.create_histogram(
            name=AGENT_EXECUTION_DURATION_METRIC_NAME,
            description="The duration of agent execution",
            unit=DURATION_UNIT_NAME,
        )

        return {
            MCP_TOOL_EXECUTION_DURATION_METRIC_NAME: mcp_tool_execution_duration,
            AGENT_EXECUTION_DURATION_METRIC_NAME: agent_execution_duration,
        }

    @staticmethod
    def _create_client_request_instruments(
        meter: Meter,
    ) -> Dict[str, Any]:
        client_request_duration = meter.create_histogram(
            name=RASA_CLIENT_REQUEST_DURATION_METRIC_NAME,
            description="The duration of the rasa client request",
            unit=DURATION_UNIT_NAME,
        )

        client_request_body_size = meter.create_histogram(
            name=RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME,
            description="The rasa client request's body size",
            unit="byte",
        )

        return {
            RASA_CLIENT_REQUEST_DURATION_METRIC_NAME: client_request_duration,
            RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME: client_request_body_size,
        }

    @staticmethod
    def _create_mcp_agent_llm_instruments(meter: Meter) -> Dict[str, Any]:
        """Create instruments for MCP agent LLM calls."""
        mcp_agent_llm_cpu_usage = meter.create_histogram(
            name=MCP_AGENT_LLM_CPU_USAGE_METRIC_NAME,
            description="CPU usage during MCP agent LLM calls",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        mcp_agent_llm_memory_usage = meter.create_histogram(
            name=MCP_AGENT_LLM_MEMORY_USAGE_METRIC_NAME,
            description="Memory usage during MCP agent LLM calls",
            unit=LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME,
        )

        mcp_agent_llm_prompt_token_usage = meter.create_histogram(
            name=MCP_AGENT_LLM_PROMPT_TOKEN_USAGE_METRIC_NAME,
            description="Prompt token usage for MCP agent LLM calls",
            unit="token",
        )

        mcp_agent_llm_response_duration = meter.create_histogram(
            name=MCP_AGENT_LLM_RESPONSE_DURATION_METRIC_NAME,
            description="Duration of MCP agent LLM calls",
            unit=DURATION_UNIT_NAME,
        )

        return {
            MCP_AGENT_LLM_CPU_USAGE_METRIC_NAME: mcp_agent_llm_cpu_usage,
            MCP_AGENT_LLM_MEMORY_USAGE_METRIC_NAME: mcp_agent_llm_memory_usage,
            MCP_AGENT_LLM_PROMPT_TOKEN_USAGE_METRIC_NAME: (
                mcp_agent_llm_prompt_token_usage
            ),
            MCP_AGENT_LLM_RESPONSE_DURATION_METRIC_NAME: (
                mcp_agent_llm_response_duration
            ),
        }
