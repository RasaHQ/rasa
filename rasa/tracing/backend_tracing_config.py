from __future__ import annotations

import abc
import base64
import os
from typing import Any, Dict, Optional, Text

import grpc
import structlog
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
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
    ENDPOINTS_OTLP_BACKEND_TYPE,
    ENDPOINTS_ROOT_CERTIFICATES_KEY,
    ENDPOINTS_TRACING_KEY,
    ENDPOINTS_TRACING_SERVICE_NAME_KEY,
)
from rasa.tracing.instrumentation import instrumentation
from rasa.utils.endpoints import (
    EndpointConfig,
    read_backend_tracing_configuration,
)

TRACING_SERVICE_NAME = os.environ.get("TRACING_SERVICE_NAME", "rasa")

structlogger = structlog.get_logger()


def configure_backend_tracing(endpoints_file: str) -> None:
    """Configure tracing functionality."""
    tracer_provider = _get_tracer_provider(endpoints_file)

    if tracer_provider is None:
        return None

    classes_to_instrument = _collect_classes_to_instrument()

    instrumentation.instrument(
        tracer_provider=tracer_provider,
        agent_class=Agent,
        processor_class=MessageProcessor,
        tracker_store_class=TrackerStore,
        graph_node_class=GraphNode,
        graph_trainer_class=GraphTrainer,
        llm_command_generator_class=LLMCommandGenerator,
        command_subclasses=classes_to_instrument["command_subclasses"],
        contextual_response_rephraser_class=classes_to_instrument[
            "contextual_response_rephraser_class"
        ],
        policy_subclasses=classes_to_instrument["policy_subclasses"],
        vector_store_subclasses=classes_to_instrument["vector_store_subclasses"],
        nlu_command_adapter_class=NLUCommandAdapter,
        endpoint_config_class=EndpointConfig,
        grpc_custom_action_executor_class=GRPCCustomActionExecutor,
        single_step_llm_command_generator_class=SingleStepLLMCommandGenerator,
        compact_llm_command_generator_class=CompactLLMCommandGenerator,
        search_ready_llm_command_generator_class=SearchReadyLLMCommandGenerator,
        multi_step_llm_command_generator_class=MultiStepLLMCommandGenerator,
        custom_action_executor_subclasses=classes_to_instrument[
            "custom_action_executor_subclasses"
        ],
        flow_retrieval_class=FlowRetrieval,
        subagent_classes=classes_to_instrument["agent_classes"],
    )


def _collect_classes_to_instrument() -> Dict[str, Any]:
    """Collect all classes that need to be instrumented for tracing."""
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

    agent_classes = [MCPOpenAgent, MCPTaskAgent, A2AAgent]

    return {
        "command_subclasses": command_subclasses,
        "policy_subclasses": policy_subclasses,
        "vector_store_subclasses": vector_store_subclasses,
        "custom_action_executor_subclasses": custom_action_executor_subclasses,
        "agent_classes": agent_classes,
        "contextual_response_rephraser_class": ContextualResponseRephraser,
    }


def _get_tracer_provider(endpoints_file: Text) -> Optional[TracerProvider]:
    """Configure tracing backend.

    When a known tracing backend is defined in the endpoints file, this
    function will configure the tracing infrastructure. When no or an unknown
    tracing backend is defined, this function does noth
    """
    tracing_config = read_backend_tracing_configuration(
        endpoints_file, ENDPOINTS_TRACING_KEY
    )

    if not tracing_config:
        structlogger.info(
            "endpoint.read.no_tracing_config",
            filename=os.path.abspath(endpoints_file),
            event_info=(
                f"No endpoint for tracing type available in {endpoints_file}, "
                f"tracing will not be configured."
            ),
        )
        return None

    tracer_provider: Optional[TracerProvider] = None
    if tracing_config.type == "jaeger":
        tracer_provider = JaegerTracerConfigurer.configure_from_endpoint_config(
            tracing_config
        )
    elif tracing_config.type == ENDPOINTS_OTLP_BACKEND_TYPE:
        tracer_provider = OTLPCollectorConfigurer.configure_from_endpoint_config(
            tracing_config
        )
    else:
        structlogger.warning(
            "endpoint.read.unknown_tracing_type",
            filename=os.path.abspath(endpoints_file),
            event_info=(
                f"Unknown tracing type {tracing_config.type} read from "
                f"{endpoints_file}, ignoring."
            ),
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
    def configure_from_endpoint_config(cls, config: EndpointConfig) -> TracerProvider:
        """Configure tracing.

        This abstract method should be implemented by all concrete `TracerConfigurer`s.
        It shall read the configuration from the supplied argument, configure all
        necessary infrastructure for tracing, and return the `TracerProvider` to be
        used for tracing purposes.

        :param config: The configuration to be read for configuring tracing.
        :return: The configured `TracerProvider`.
        """


class JaegerTracerConfigurer(TracerConfigurer):
    """The `TracerConfigurer` for a Jaeger backend.

    This class maintains backward compatibility with the old Jaeger configuration
    format while internally using OTLP to avoid protobuf compatibility issues.
    """

    @classmethod
    def configure_from_endpoint_config(cls, config: EndpointConfig) -> TracerProvider:
        """Configure tracing for Jaeger using OTLP under the hood.

        This maintains backward compatibility with the old Jaeger configuration format
        while using OTLP internally to avoid protobuf compatibility issues.

        :param config: The configuration to be read for configuring tracing.
        :return: The configured `TracerProvider`.
        """
        jaeger_config = cls._extract_config(config)
        otlp_endpoint = cls._build_otlp_endpoint(jaeger_config)
        otlp_exporter = cls._create_otlp_exporter(jaeger_config, otlp_endpoint)
        provider = cls._create_tracer_provider(config)

        structlogger.info(
            "endpoint.read.jaeger_config_configured",
            event_info=(
                f"Registered {config.type} endpoint for tracing using OTLP. "
                f"Traces will be exported to {otlp_endpoint}."
            ),
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        return provider

    @classmethod
    def _extract_config(cls, config: EndpointConfig) -> Dict[str, Any]:
        """Extract Jaeger configuration parameters."""
        return {
            "agent_host_name": config.kwargs.get("host", "localhost"),
            "agent_port": config.kwargs.get("port", 6831),
            "username": config.kwargs.get("username"),
            "password": config.kwargs.get("password"),
        }

    @classmethod
    def _build_otlp_endpoint(cls, jaeger_config: Dict[str, Any]) -> str:
        """Build OTLP endpoint URL from Jaeger configuration."""
        host = jaeger_config["agent_host_name"]
        port = jaeger_config.get("agent_port", 4317)
        return f"http://{host}:{port}"

    @classmethod
    def _create_otlp_exporter(
        cls, jaeger_config: Dict[str, Any], otlp_endpoint: str
    ) -> OTLPSpanExporter:
        """Create OTLP exporter with Jaeger-compatible configuration."""
        headers = cls._build_headers(jaeger_config)
        return OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,  # Jaeger typically runs without TLS in development
            headers=headers,
        )

    @classmethod
    def _create_tracer_provider(cls, config: EndpointConfig) -> TracerProvider:
        """Create TracerProvider with service name from configuration."""
        service_name = config.kwargs.get(
            ENDPOINTS_TRACING_SERVICE_NAME_KEY, TRACING_SERVICE_NAME
        )
        return TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))

    @classmethod
    def _build_headers(cls, jaeger_config: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Build OTLP headers from Jaeger authentication config."""
        username = jaeger_config.get("username")
        password = jaeger_config.get("password")

        if not username or not password:
            return None

        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {credentials}"}


class OTLPCollectorConfigurer(TracerConfigurer):
    """The `TracerConfigurer` for an OTLP collector backend."""

    @classmethod
    def configure_from_endpoint_config(cls, config: EndpointConfig) -> TracerProvider:
        """Configure tracing for OTLP Collector.

        This will read the OTLP collector-specific configuration from the
        `EndpointConfig` and create a corresponding `TracerProvider` that exports to
        the given OTLP collector.
        Currently, this only supports insecure connections via gRPC.

        :param config: The configuration to be read for configuring tracing.
        :return: The configured `TracerProvider`.
        """
        provider = cls._create_tracer_provider(config)
        insecure = config.kwargs.get(ENDPOINTS_INSECURE_KEY)
        credentials = _get_credentials(config, insecure)
        otlp_exporter = cls._create_otlp_exporter(config, insecure, credentials)

        structlogger.info(
            "endpoint.read.otlp_collector_config_configured",
            event_info=(
                f"Registered {config.type} endpoint for tracing. "
                f"Traces will be exported to {config.kwargs[ENDPOINTS_ENDPOINT_KEY]}"
            ),
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        return provider

    @classmethod
    def _create_tracer_provider(cls, config: EndpointConfig) -> TracerProvider:
        """Create TracerProvider with service name from configuration."""
        service_name = config.kwargs.get(
            ENDPOINTS_TRACING_SERVICE_NAME_KEY, TRACING_SERVICE_NAME
        )
        return TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))

    @classmethod
    def _create_otlp_exporter(
        cls,
        config: EndpointConfig,
        insecure: Optional[bool],
        credentials: Optional[grpc.ChannelCredentials],
    ) -> OTLPSpanExporter:
        """Create OTLP exporter from configuration."""
        return OTLPSpanExporter(
            endpoint=config.kwargs[ENDPOINTS_ENDPOINT_KEY],
            insecure=insecure,
            credentials=credentials,
        )


def _get_credentials(
    config: EndpointConfig, insecure: Optional[bool]
) -> Optional[grpc.ChannelCredentials]:
    """Get gRPC credentials from configuration."""
    credentials = None
    if not insecure and ENDPOINTS_ROOT_CERTIFICATES_KEY in config.kwargs:
        with open(config.kwargs.get(ENDPOINTS_ROOT_CERTIFICATES_KEY), "rb") as f:
            root_cert = f.read()
        credentials = grpc.ssl_channel_credentials(root_certificates=root_cert)
    return credentials
