from __future__ import annotations

import abc
import logging
import os
from typing import Any, Dict, Optional, Text, ClassVar

import grpc
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import get_meter_provider
from opentelemetry.sdk.metrics import Meter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from rasa.core.agent import Agent
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import TrackerStore
from rasa.dialogue_understanding.commands import (
    Command,
    FreeFormAnswerCommand,
)
from rasa.dialogue_understanding.generator.llm_command_generator import (
    LLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.engine.graph import GraphNode
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.tracing.constants import (
    LLM_COMMAND_GENERATOR_CPU_USAGE_INSTRUMENT_NAME,
    LLM_COMMAND_GENERATOR_MEMORY_USAGE_INSTRUMENT_NAME,
    LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_INSTRUMENT_NAME,
    LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    INTENTLESS_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    ACTION_SERVER_RASA_CLIENT_REQUEST_DURATION_INSTRUMENT_NAME,
    ACTION_SERVER_RASA_CLIENT_REQUEST_BODY_SIZE_INSTRUMENT_NAME,
)
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

from rasa.tracing.instrumentation import instrumentation
from rasa.utils.singleton import Singleton

TRACING_SERVICE_NAME = os.environ.get("TRACING_SERVICE_NAME", "rasa")

ENDPOINTS_TRACING_KEY = "tracing"

logger = logging.getLogger(__name__)


def configure_tracing(tracer_provider: Optional[TracerProvider]) -> None:
    """Configure tracing functionality.

    When a tracing backend is defined, this function will
    instrument all methods that shall be traced.
    If no tracing backend is defined, no tracing is configured.

    :param tracer_provider: The `TracingProvider` to be used for tracing
    """
    if tracer_provider is None:
        return None

    from rasa.core.information_retrieval.information_retrieval import (
        InformationRetrieval,
    )
    from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
    from rasa.core.policies.policy import Policy
    from rasa.engine.recipes.default_components import DEFAULT_COMPONENTS

    command_subclasses = [subclass for subclass in Command.__subclasses__()] + [
        subclass for subclass in FreeFormAnswerCommand.__subclasses__()
    ]

    policy_subclasses = [
        policy_class
        for policy_class in DEFAULT_COMPONENTS
        if issubclass(policy_class, Policy)
    ]

    vector_store_subclasses = [
        vector_store_class
        for vector_store_class in InformationRetrieval.__subclasses__()
    ]

    instrumentation.instrument(
        tracer_provider=tracer_provider,
        agent_class=Agent,
        processor_class=MessageProcessor,
        tracker_store_class=TrackerStore,
        graph_node_class=GraphNode,
        graph_trainer_class=GraphTrainer,
        llm_command_generator_class=LLMCommandGenerator,
        command_subclasses=command_subclasses,
        contextual_response_rephraser_class=ContextualResponseRephraser,
        policy_subclasses=policy_subclasses,
        vector_store_subclasses=vector_store_subclasses,
        nlu_command_adapter_class=NLUCommandAdapter,
    )


def get_tracer_provider(endpoints_file: Text) -> Optional[TracerProvider]:
    """Configure tracing backend.

    When a known tracing backend is defined in the endpoints file, this
    function will configure the tracing infrastructure. When no or an unknown
    tracing backend is defined, this function does nothing.

    :param endpoints_file: The configuration file containing information about the
        tracing backend.
    :return: The `TracingProvider` to be used for all subsequent tracing.
    """
    cfg = read_endpoint_config(endpoints_file, ENDPOINTS_TRACING_KEY)

    if not cfg:
        logger.info(
            f"No endpoint for tracing type available in {endpoints_file},"
            f"tracing will not be configured."
        )
        return None
    if cfg.type == "jaeger":
        tracer_provider = JaegerTracerConfigurer.configure_from_endpoint_config(cfg)
    elif cfg.type == "otlp":
        tracer_provider = OTLPCollectorConfigurer.configure_from_endpoint_config(cfg)
    else:
        logger.warning(
            f"Unknown tracing type {cfg.type} read from {endpoints_file}, ignoring."
        )
        return None

    return tracer_provider


class TracerConfigurer(abc.ABC):
    """Abstract superclass for tracing configuration.

    `TracerConfigurer` is the abstract superclass from which all configurers
    for different supported backends should inherit.
    """

    @classmethod
    @abc.abstractmethod
    def configure_from_endpoint_config(cls, cfg: EndpointConfig) -> TracerProvider:
        """Configure tracing.

        This abstract method should be implemented by all concrete `TracerConfigurer`s.
        It shall read the configuration from the supplied argument, configure all
        necessary infrastructure for tracing, and return the `TracerProvider` to be
        used for tracing purposes.

        :param cfg: The configuration to be read for configuring tracing.
        :return: The configured `TracerProvider`.
        """


class JaegerTracerConfigurer(TracerConfigurer):
    """The `TracerConfigurer` for a Jaeger backend."""

    @classmethod
    def configure_from_endpoint_config(cls, cfg: EndpointConfig) -> TracerProvider:
        """Configure tracing for Jaeger.

        This will read the Jaeger-specific configuration from the `EndpointConfig` and
        create a corresponding `TracerProvider` that exports to the given Jaeger
        backend.

        :param cfg: The configuration to be read for configuring tracing.
        :return: The configured `TracerProvider`.
        """
        provider = TracerProvider(
            resource=Resource.create(
                {SERVICE_NAME: cfg.kwargs.get("service_name", TRACING_SERVICE_NAME)}
            )
        )

        jaeger_exporter = JaegerExporter(
            **cls._extract_config(cfg), udp_split_oversized_batches=True
        )
        logger.info(
            f"Registered {cfg.type} endpoint for tracing. Traces will be exported to"
            f" {jaeger_exporter.agent_host_name}:{jaeger_exporter.agent_port}"
        )
        provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

        return provider

    @classmethod
    def _extract_config(cls, cfg: EndpointConfig) -> Dict[str, Any]:
        return {
            "agent_host_name": (cfg.kwargs.get("host", "localhost")),
            "agent_port": (cfg.kwargs.get("port", 6831)),
            "username": cfg.kwargs.get("username"),
            "password": cfg.kwargs.get("password"),
        }


class OTLPCollectorConfigurer(TracerConfigurer):
    """The `TracerConfigurer` for an OTLP collector backend."""

    @classmethod
    def configure_from_endpoint_config(cls, cfg: EndpointConfig) -> TracerProvider:
        """Configure tracing for Jaeger.

        This will read the OTLP collector-specific configuration from the
        `EndpointConfig` and create a corresponding `TracerProvider` that exports to
        the given OTLP collector.
        Currently, this only supports insecure connections via gRPC.

        :param cfg: The configuration to be read for configuring tracing.
        :return: The configured `TracerProvider`.
        """
        provider = TracerProvider(
            resource=Resource.create(
                {SERVICE_NAME: cfg.kwargs.get("service_name", TRACING_SERVICE_NAME)}
            )
        )

        insecure = cfg.kwargs.get("insecure")

        credentials = cls._get_credentials(cfg, insecure)

        otlp_exporter = OTLPSpanExporter(
            endpoint=cfg.kwargs["endpoint"],
            insecure=insecure,
            credentials=credentials,
        )
        logger.info(
            f"Registered {cfg.type} endpoint for tracing."
            f"Traces will be exported to {cfg.kwargs['endpoint']}"
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        return provider

    @classmethod
    def _get_credentials(
        cls, cfg: EndpointConfig, insecure: bool
    ) -> Optional[grpc.ChannelCredentials]:
        credentials = None
        if not insecure and "root_certificates" in cfg.kwargs:
            with open(cfg.kwargs.get("root_certificates"), "rb") as f:
                root_cert = f.read()
            credentials = grpc.ssl_channel_credentials(root_certificates=root_cert)
        return credentials


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
            **self._create_llm_response_duration_instruments(meter),
            **self._create_action_server_client_request_instruments(meter),
        }

        self.instruments.update(instruments)

    def get_instrument(self, name: str) -> Any:
        """Get the instrument mapped to the provided name."""
        return self.instruments.get(name)

    @staticmethod
    def _create_llm_command_generator_instruments(meter: Meter) -> Dict[str, Any]:
        llm_command_generator_cpu_usage = meter.create_histogram(
            name=LLM_COMMAND_GENERATOR_CPU_USAGE_INSTRUMENT_NAME,
            description="cpu percentage for LLMCommandGenerator",
            unit="%",
        )

        llm_command_generator_memory_usage = meter.create_histogram(
            name=LLM_COMMAND_GENERATOR_MEMORY_USAGE_INSTRUMENT_NAME,
            description="RAM memory usage for LLMCommandGenerator",
            unit="%",
        )

        llm_command_generator_prompt_token_usage = meter.create_histogram(
            name=LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_INSTRUMENT_NAME,
            description="LLMCommandGenerator prompt token length",
            unit="1",
        )

        llm_response_duration_llm_command_generator = meter.create_histogram(
            name=LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
            description="The duration of LLMCommandGenerator's LLM call",
            unit="ms",
        )

        return {
            LLM_COMMAND_GENERATOR_CPU_USAGE_INSTRUMENT_NAME: llm_command_generator_cpu_usage,  # noqa: E501
            LLM_COMMAND_GENERATOR_MEMORY_USAGE_INSTRUMENT_NAME: llm_command_generator_memory_usage,  # noqa: E501
            LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_INSTRUMENT_NAME: llm_command_generator_prompt_token_usage,  # noqa: E501
            LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_INSTRUMENT_NAME: llm_response_duration_llm_command_generator,  # noqa: E501
        }

    @staticmethod
    def _create_llm_response_duration_instruments(meter: Meter) -> Dict[str, Any]:
        llm_response_duration_enterprise_search = meter.create_histogram(
            name=ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
            description="The duration of EnterpriseSearchPolicy's LLM call",
            unit="ms",
        )

        llm_response_duration_intentless = meter.create_histogram(
            name=INTENTLESS_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
            description="The duration of IntentlessPolicy's LLM call",
            unit="ms",
        )

        llm_response_duration_contextual_nlg = meter.create_histogram(
            name=CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
            description="The duration of ContextualResponseRephraser's LLM call",
            unit="ms",
        )

        return {
            ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME: llm_response_duration_enterprise_search,  # noqa: E501
            INTENTLESS_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME: llm_response_duration_intentless,  # noqa: E501
            CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_INSTRUMENT_NAME: llm_response_duration_contextual_nlg,  # noqa: E501
        }

    @staticmethod
    def _create_action_server_client_request_instruments(
        meter: Meter,
    ) -> Dict[str, Any]:
        action_server_client_request_duration = meter.create_histogram(
            name=ACTION_SERVER_RASA_CLIENT_REQUEST_DURATION_INSTRUMENT_NAME,
            description="The duration of the rasa client request to the action server",
            unit="ms",
        )

        action_server_client_request_body_size = meter.create_histogram(
            name=ACTION_SERVER_RASA_CLIENT_REQUEST_BODY_SIZE_INSTRUMENT_NAME,
            description="The rasa client request's body size to the action server",
            unit="byte",
        )

        return {
            ACTION_SERVER_RASA_CLIENT_REQUEST_DURATION_INSTRUMENT_NAME: action_server_client_request_duration,  # noqa: E501
            ACTION_SERVER_RASA_CLIENT_REQUEST_BODY_SIZE_INSTRUMENT_NAME: action_server_client_request_body_size,  # noqa: E501
        }
