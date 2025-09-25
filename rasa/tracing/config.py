from __future__ import annotations

import abc
import logging
import os
from typing import Any, Dict, Optional, Text

import grpc
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from rasa.agents.protocol.a2a.a2a_agent import A2AAgent
from rasa.agents.protocol.mcp.mcp_open_agent import MCPOpenAgent
from rasa.agents.protocol.mcp.mcp_task_agent import MCPTaskAgent
from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
    RetryCustomActionExecutor,
)
from rasa.core.actions.grpc_custom_action_executor import GRPCCustomActionExecutor
from rasa.core.agent import Agent
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.dialogue_understanding.commands import Command, FreeFormAnswerCommand
from rasa.dialogue_understanding.generator import (
    CompactLLMCommandGenerator,
    LLMCommandGenerator,
    MultiStepLLMCommandGenerator,
    SearchReadyLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.engine.graph import GraphNode
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.tracing.constants import (
    ENDPOINTS_ENDPOINT_KEY,
    ENDPOINTS_INSECURE_KEY,
    ENDPOINTS_METRICS_KEY,
    ENDPOINTS_OTLP_BACKEND_TYPE,
    ENDPOINTS_ROOT_CERTIFICATES_KEY,
    ENDPOINTS_TRACING_KEY,
    ENDPOINTS_TRACING_SERVICE_NAME_KEY,
)
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

TRACING_SERVICE_NAME = os.environ.get("TRACING_SERVICE_NAME", "rasa")

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

    custom_action_executor_subclasses = []
    for custom_action_executor_class in CustomActionExecutor.__subclasses__():
        if custom_action_executor_class != RetryCustomActionExecutor:
            custom_action_executor_subclasses.append(custom_action_executor_class)

    # All agent classes to instrument (protocol agent classes)
    agent_classes = [MCPOpenAgent, MCPTaskAgent, A2AAgent]

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
        endpoint_config_class=EndpointConfig,
        grpc_custom_action_executor_class=GRPCCustomActionExecutor,
        single_step_llm_command_generator_class=SingleStepLLMCommandGenerator,
        compact_llm_command_generator_class=CompactLLMCommandGenerator,
        search_ready_llm_command_generator_class=SearchReadyLLMCommandGenerator,
        multi_step_llm_command_generator_class=MultiStepLLMCommandGenerator,
        custom_action_executor_subclasses=custom_action_executor_subclasses,
        flow_retrieval_class=FlowRetrieval,
        subagent_classes=agent_classes,
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
            f"No endpoint for tracing type available in {endpoints_file}, "
            f"tracing will not be configured."
        )
        return None
    if cfg.type == "jaeger":
        tracer_provider = JaegerTracerConfigurer.configure_from_endpoint_config(cfg)
    elif cfg.type == ENDPOINTS_OTLP_BACKEND_TYPE:
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
    """The `TracerConfigurer` for a Jaeger backend.

    This class maintains backward compatibility with the old Jaeger configuration format
    while internally using OTLP to avoid protobuf compatibility issues.
    """

    @classmethod
    def configure_from_endpoint_config(cls, cfg: EndpointConfig) -> TracerProvider:
        """Configure tracing for Jaeger using OTLP under the hood.

        This maintains backward compatibility with the old Jaeger configuration format
        while using OTLP internally to avoid protobuf compatibility issues.

        :param cfg: The configuration to be read for configuring tracing.
        :return: The configured `TracerProvider`.
        """
        # Extract Jaeger-specific configuration
        jaeger_config = cls._extract_config(cfg)

        # Map Jaeger configuration to OTLP endpoint
        otlp_endpoint = f"http://{jaeger_config['agent_host_name']}:{jaeger_config.get('agent_port', 4317)}"  # noqa: E501

        # Create OTLP exporter with Jaeger-compatible configuration
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,  # Jaeger typically runs without TLS in development
            headers=cls._build_headers(jaeger_config),
        )

        provider = TracerProvider(
            resource=Resource.create(
                {
                    SERVICE_NAME: cfg.kwargs.get(
                        ENDPOINTS_TRACING_SERVICE_NAME_KEY, TRACING_SERVICE_NAME
                    )
                }
            )
        )

        logger.info(
            f"Registered {cfg.type} endpoint for tracing using OTLP. "
            f"Traces will be exported to {otlp_endpoint}."
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        return provider

    @classmethod
    def _extract_config(cls, cfg: EndpointConfig) -> Dict[str, Any]:
        """Extract Jaeger configuration parameters."""
        return {
            "agent_host_name": (cfg.kwargs.get("host", "localhost")),
            "agent_port": (cfg.kwargs.get("port", 6831)),
            "username": cfg.kwargs.get("username"),
            "password": cfg.kwargs.get("password"),
        }

    @classmethod
    def _build_headers(cls, jaeger_config: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Build OTLP headers from Jaeger authentication config."""
        headers = {}
        if jaeger_config.get("username") and jaeger_config.get("password"):
            import base64

            credentials = base64.b64encode(
                f"{jaeger_config['username']}:{jaeger_config['password']}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        return headers if headers else None


class OTLPCollectorConfigurer(TracerConfigurer):
    """The `TracerConfigurer` for an OTLP collector backend."""

    @classmethod
    def configure_from_endpoint_config(cls, cfg: EndpointConfig) -> TracerProvider:
        """Configure tracing for OTLP Collector.

        This will read the OTLP collector-specific configuration from the
        `EndpointConfig` and create a corresponding `TracerProvider` that exports to
        the given OTLP collector.
        Currently, this only supports insecure connections via gRPC.

        :param cfg: The configuration to be read for configuring tracing.
        :return: The configured `TracerProvider`.
        """
        provider = TracerProvider(
            resource=Resource.create(
                {
                    SERVICE_NAME: cfg.kwargs.get(
                        ENDPOINTS_TRACING_SERVICE_NAME_KEY, TRACING_SERVICE_NAME
                    )
                }
            )
        )

        insecure = cfg.kwargs.get(ENDPOINTS_INSECURE_KEY)

        credentials = cls._get_credentials(cfg, insecure)

        otlp_exporter = OTLPSpanExporter(
            endpoint=cfg.kwargs[ENDPOINTS_ENDPOINT_KEY],
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
        return _get_credentials(cfg, insecure)


class OTLPMetricConfigurer:
    """The metric configurer for the OTLP Collector backend."""

    @classmethod
    def configure_from_endpoint_config(
        cls, cfg: EndpointConfig
    ) -> Optional[OTLPMetricExporter]:
        """Configure metrics for OTLP Collector.

        This will read the OTLP collector-specific configuration from the
        `EndpointConfig` and create a corresponding `OTLPMetricExporter` that exports to
        the given OTLP collector.

        :param cfg: The configuration to be read for configuring metrics.
        :return: The configured `OTLPMetricExporter`.
        """
        insecure = cfg.kwargs.get(ENDPOINTS_INSECURE_KEY)

        credentials = cls._get_credentials(cfg, insecure)

        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=cfg.kwargs[ENDPOINTS_ENDPOINT_KEY],
            insecure=insecure,
            credentials=credentials,
        )
        logger.info(
            f"Registered '{cfg.type}' endpoint for metrics. "
            f"Metrics will be exported to {cfg.kwargs['endpoint']}."
        )

        return otlp_metric_exporter

    @classmethod
    def _get_credentials(
        cls, cfg: EndpointConfig, insecure: bool
    ) -> Optional[grpc.ChannelCredentials]:
        return _get_credentials(cfg, insecure)


def configure_metrics(endpoints_file: str) -> None:
    """Configure metrics export for OTLP Collector.

    :param endpoints_file: The configuration file containing information about the
        metrics backend.
    :return: None.
    """
    cfg = read_endpoint_config(endpoints_file, ENDPOINTS_METRICS_KEY)

    if not cfg:
        logger.debug(
            "The OTLP Collector has not been configured to collect "
            "metrics. Skipping."
        )
        return None

    if cfg.type != ENDPOINTS_OTLP_BACKEND_TYPE:
        logger.warning(
            f"Unknown metrics backend type '{cfg.type}' "
            f"read from '{endpoints_file}', ignoring."
        )
        return None

    otlp_exporter = OTLPMetricConfigurer.configure_from_endpoint_config(cfg)
    metric_reader = PeriodicExportingMetricReader(otlp_exporter)
    set_meter_provider(
        MeterProvider(
            metric_readers=[metric_reader],
            resource=Resource.create(
                {
                    SERVICE_NAME: cfg.kwargs.get(
                        ENDPOINTS_TRACING_SERVICE_NAME_KEY, TRACING_SERVICE_NAME
                    )
                }
            ),
        )
    )

    MetricInstrumentProvider().register_instruments()


def _get_credentials(
    cfg: EndpointConfig, insecure: bool
) -> Optional[grpc.ChannelCredentials]:
    credentials = None
    if not insecure and ENDPOINTS_ROOT_CERTIFICATES_KEY in cfg.kwargs:
        with open(cfg.kwargs.get(ENDPOINTS_ROOT_CERTIFICATES_KEY), "rb") as f:
            root_cert = f.read()
        credentials = grpc.ssl_channel_credentials(root_certificates=root_cert)
    return credentials
