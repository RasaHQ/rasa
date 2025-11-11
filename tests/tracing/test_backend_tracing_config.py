import textwrap
import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import grpc

from rasa.tracing.backend_tracing_config import (
    JaegerTracerConfigurer,
    OTLPCollectorConfigurer,
    _collect_classes_to_instrument,
    _get_credentials,
    _get_tracer_provider,
    configure_backend_tracing,
)
from rasa.utils.endpoints import EndpointConfig
from tests.conftest import wait
from tests.tracing.conftest import (
    TRACING_TESTS_FIXTURES_DIRECTORY,
    CapturingTestSpanExporter,
)


def test_jaeger_config_correctly_extracted() -> None:
    """Test that Jaeger config is correctly extracted from EndpointConfig."""
    config = EndpointConfig(
        host="hostname",
        port=1234,
        username="user",
        password="password",
    )

    extracted = JaegerTracerConfigurer._extract_config(config)

    assert extracted["agent_host_name"] == config.kwargs["host"]
    assert extracted["agent_port"] == config.kwargs["port"]
    assert extracted["username"] == config.kwargs["username"]
    assert extracted["password"] == config.kwargs["password"]


def test_jaeger_config_sets_defaults() -> None:
    """Test that Jaeger config sets default values when not provided."""
    extracted = JaegerTracerConfigurer._extract_config(EndpointConfig())

    assert extracted["agent_host_name"] == "localhost"
    assert extracted["agent_port"] == 6831
    assert extracted["username"] is None
    assert extracted["password"] is None


def test_get_tracer_provider_otlp_collector(
    grpc_server: grpc.Server,
    span_exporter: CapturingTestSpanExporter,
    result_available_event: threading.Event,
) -> None:
    """Test that OTLP collector tracer provider is correctly configured."""
    endpoints_file = str(TRACING_TESTS_FIXTURES_DIRECTORY / "otlp_endpoints.yml")

    tracer_provider = _get_tracer_provider(endpoints_file)
    assert tracer_provider is not None

    tracer = tracer_provider.get_tracer("foo")

    with tracer.start_as_current_span("otlp_test_span"):
        pass

    tracer_provider.force_flush()

    wait(
        lambda: span_exporter.spans is not None,
        result_available_event=result_available_event,
        timeout_seconds=15,
    )

    spans = span_exporter.spans

    assert spans is not None
    assert len(spans[0].scope_spans[0].spans) == 1
    assert spans[0].scope_spans[0].spans[0].name == "otlp_test_span"


def test_get_tracer_provider_tls_otlp_collector(
    secured_grpc_server: grpc.Server,
    span_exporter: CapturingTestSpanExporter,
    result_available_event: threading.Event,
) -> None:
    """Test that TLS OTLP collector tracer provider is correctly configured."""
    endpoints_file = str(TRACING_TESTS_FIXTURES_DIRECTORY / "otlp_endpoints_tls.yml")

    tracer_provider = _get_tracer_provider(endpoints_file)
    assert tracer_provider is not None

    tracer = tracer_provider.get_tracer("foo")

    with tracer.start_as_current_span("otlp_test_span"):
        pass

    tracer_provider.force_flush()

    wait(
        lambda: span_exporter.spans is not None,
        result_available_event=result_available_event,
        timeout_seconds=15,
    )

    spans = span_exporter.spans

    assert spans is not None
    assert len(spans[0].scope_spans[0].spans) == 1
    assert spans[0].scope_spans[0].spans[0].name == "otlp_test_span"


def test_get_tracer_provider_jaeger(
    grpc_server: grpc.Server,
    span_exporter: CapturingTestSpanExporter,
    result_available_event: threading.Event,
) -> None:
    """Test that Jaeger tracer provider is correctly configured."""
    endpoints_file = str(TRACING_TESTS_FIXTURES_DIRECTORY / "jaeger_endpoints.yml")

    tracer_provider = _get_tracer_provider(endpoints_file)
    assert tracer_provider is not None

    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span("jaeger_test_span"):
        pass

    tracer_provider.force_flush()

    wait(
        lambda: span_exporter.spans is not None,
        result_available_event=result_available_event,
        timeout_seconds=15,
    )

    spans = span_exporter.spans
    assert spans is not None
    assert len(spans[0].scope_spans[0].spans) == 1
    assert spans[0].scope_spans[0].spans[0].name == "jaeger_test_span"


def test_get_tracer_provider_unknown_type(tmp_path: Path) -> None:
    """Test that get_tracer_provider returns None for unknown tracing type."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        """
tracing:
    type: unknown_type
    endpoint: http://localhost:4317
"""
    )

    tracer_provider = _get_tracer_provider(str(endpoints_file))
    assert tracer_provider is None


def test_get_tracer_provider_no_config(tmp_path: Path) -> None:
    """Test that get_tracer_provider returns None when no config exists."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        """
action_endpoint:
    url: "http://localhost:5056/webhook"
"""
    )

    tracer_provider = _get_tracer_provider(str(endpoints_file))
    assert tracer_provider is None


def test_jaeger_tracer_configurer_build_otlp_endpoint() -> None:
    """Test that Jaeger configurer builds correct OTLP endpoint."""
    jaeger_config = {
        "agent_host_name": "localhost",
        "agent_port": 6831,
    }
    endpoint = JaegerTracerConfigurer._build_otlp_endpoint(jaeger_config)
    assert endpoint == "http://localhost:6831"


def test_jaeger_tracer_configurer_build_otlp_endpoint_default_port() -> None:
    """Test that Jaeger configurer uses default port when not specified."""
    jaeger_config = {
        "agent_host_name": "localhost",
    }
    endpoint = JaegerTracerConfigurer._build_otlp_endpoint(jaeger_config)
    assert endpoint == "http://localhost:4317"


def test_jaeger_tracer_configurer_build_headers_with_auth() -> None:
    """Test that Jaeger configurer builds headers with authentication."""
    jaeger_config = {
        "username": "test_user",
        "password": "test_password",
    }
    headers = JaegerTracerConfigurer._build_headers(jaeger_config)
    assert headers is not None
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("Basic ")


def test_jaeger_tracer_configurer_build_headers_without_auth() -> None:
    """Test that Jaeger configurer returns None when no auth provided."""
    jaeger_config = {}
    headers = JaegerTracerConfigurer._build_headers(jaeger_config)
    assert headers is None


def test_jaeger_tracer_configurer_create_tracer_provider() -> None:
    """Test that Jaeger configurer creates tracer provider with service name."""
    config = EndpointConfig(type="jaeger", service_name="test-service")
    provider = JaegerTracerConfigurer._create_tracer_provider(config)
    assert provider is not None


def test_otlp_collector_configurer_create_tracer_provider() -> None:
    """Test that OTLP collector configurer creates tracer provider."""
    config = EndpointConfig(
        type="otlp",
        endpoint="http://localhost:4317",
        service_name="test-service",
    )
    provider = OTLPCollectorConfigurer._create_tracer_provider(config)
    assert provider is not None


def test_otlp_collector_configurer_create_otlp_exporter() -> None:
    """Test that OTLP collector configurer creates exporter."""
    config = EndpointConfig(
        type="otlp",
        endpoint="http://localhost:4317",
    )
    exporter = OTLPCollectorConfigurer._create_otlp_exporter(config, True, None)
    assert exporter is not None


def test_get_credentials_without_tls() -> None:
    """Test that _get_credentials returns None when insecure is True."""
    config = EndpointConfig(
        type="otlp",
        endpoint="http://localhost:4317",
        insecure=True,
    )
    credentials = _get_credentials(config, True)
    assert credentials is None


def test_get_credentials_without_root_certificates() -> None:
    """Test that _get_credentials returns None when no root certificates provided."""
    config = EndpointConfig(
        type="otlp",
        endpoint="http://localhost:4317",
        insecure=False,
    )
    credentials = _get_credentials(config, False)
    assert credentials is None


def test_get_credentials_with_root_certificates(tmp_path: Path) -> None:
    """Test that _get_credentials creates SSL credentials when root certificates
    provided."""
    cert_file = tmp_path / "cert.pem"
    cert_content = b"-----BEGIN CERTIFICATE-----\nTEST CERT\n-----END CERTIFICATE-----"
    cert_file.write_bytes(cert_content)

    config = EndpointConfig(
        type="otlp",
        endpoint="http://localhost:4317",
        insecure=False,
        root_certificates=str(cert_file),
    )
    credentials = _get_credentials(config, False)
    assert credentials is not None
    assert isinstance(credentials, grpc.ChannelCredentials)


def test_jaeger_tracer_configurer_create_otlp_exporter_with_auth() -> None:
    """Test that Jaeger configurer creates OTLP exporter with authentication headers."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    jaeger_config = {
        "agent_host_name": "localhost",
        "agent_port": 6831,
        "username": "test_user",
        "password": "test_password",
    }
    otlp_endpoint = "http://localhost:6831"
    exporter = JaegerTracerConfigurer._create_otlp_exporter(
        jaeger_config, otlp_endpoint
    )
    assert exporter is not None
    assert isinstance(exporter, OTLPSpanExporter)


def test_jaeger_tracer_configurer_create_otlp_exporter_without_auth() -> None:
    """Test that Jaeger configurer creates OTLP exporter without authentication."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    jaeger_config = {
        "agent_host_name": "localhost",
        "agent_port": 6831,
    }
    otlp_endpoint = "http://localhost:6831"
    exporter = JaegerTracerConfigurer._create_otlp_exporter(
        jaeger_config, otlp_endpoint
    )
    assert exporter is not None
    assert isinstance(exporter, OTLPSpanExporter)


def test_jaeger_tracer_configurer_configure_from_endpoint_config_defaults() -> None:
    """Test that Jaeger configurer uses defaults when config is minimal."""
    config = EndpointConfig(type="jaeger")
    provider = JaegerTracerConfigurer.configure_from_endpoint_config(config)
    assert provider is not None


def test_otlp_collector_configurer_configure_from_endpoint_config_with_credentials(
    tmp_path: Path,
) -> None:
    """Test that OTLP collector configurer configures with TLS credentials."""
    cert_file = tmp_path / "cert.pem"
    cert_content = b"-----BEGIN CERTIFICATE-----\nTEST CERT\n-----END CERTIFICATE-----"
    cert_file.write_bytes(cert_content)

    config = EndpointConfig(
        type="otlp",
        endpoint="http://localhost:4317",
        insecure=False,
        root_certificates=str(cert_file),
        service_name="test-service",
    )
    provider = OTLPCollectorConfigurer.configure_from_endpoint_config(config)
    assert provider is not None


def test_collect_classes_to_instrument() -> None:
    """Test that _collect_classes_to_instrument collects all required classes."""
    classes = _collect_classes_to_instrument()

    assert "command_subclasses" in classes
    assert "policy_subclasses" in classes
    assert "vector_store_subclasses" in classes
    assert "custom_action_executor_subclasses" in classes
    assert "agent_classes" in classes
    assert "contextual_response_rephraser_class" in classes

    # Verify agent classes are included
    assert len(classes["agent_classes"]) > 0

    # Verify command subclasses is a list
    assert isinstance(classes["command_subclasses"], list)

    # Verify policy subclasses is a list
    assert isinstance(classes["policy_subclasses"], list)

    # Verify vector store subclasses is a list
    assert isinstance(classes["vector_store_subclasses"], list)

    # Verify custom action executor subclasses is a list
    assert isinstance(classes["custom_action_executor_subclasses"], list)


@patch("rasa.tracing.backend_tracing_config._get_tracer_provider")
@patch("rasa.tracing.backend_tracing_config._collect_classes_to_instrument")
@patch("rasa.tracing.backend_tracing_config.instrumentation")
def test_configure_backend_tracing_with_config(
    mock_instrumentation: MagicMock,
    mock_collect_classes: MagicMock,
    mock_get_tracer_provider: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that configure_backend_tracing configures tracing when config exists."""
    # Setup mocks
    mock_tracer_provider = Mock()
    mock_get_tracer_provider.return_value = mock_tracer_provider

    mock_classes = {
        "command_subclasses": [],
        "policy_subclasses": [],
        "vector_store_subclasses": [],
        "custom_action_executor_subclasses": [],
        "agent_classes": [],
        "contextual_response_rephraser_class": None,
    }
    mock_collect_classes.return_value = mock_classes

    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: jaeger
                host: localhost
                port: 6831
            """
        )
    )

    configure_backend_tracing(str(endpoints_file))

    mock_get_tracer_provider.assert_called_once_with(str(endpoints_file))
    mock_collect_classes.assert_called_once()
    mock_instrumentation.instrument.assert_called_once()
    # Verify tracer_provider was passed to instrument
    call_args = mock_instrumentation.instrument.call_args
    assert call_args.kwargs["tracer_provider"] == mock_tracer_provider


@patch("rasa.tracing.backend_tracing_config._get_tracer_provider")
@patch("rasa.tracing.backend_tracing_config._collect_classes_to_instrument")
@patch("rasa.tracing.backend_tracing_config.instrumentation")
def test_configure_backend_tracing_no_config(
    mock_instrumentation: MagicMock,
    mock_collect_classes: MagicMock,
    mock_get_tracer_provider: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that configure_backend_tracing does nothing when no config exists."""
    # Setup mocks
    mock_get_tracer_provider.return_value = None

    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            action_endpoint:
                url: "http://localhost:5056/webhook"
            """
        )
    )

    configure_backend_tracing(str(endpoints_file))

    mock_get_tracer_provider.assert_called_once_with(str(endpoints_file))
    # Should not collect classes or instrument when no tracer provider
    mock_collect_classes.assert_not_called()
    mock_instrumentation.instrument.assert_not_called()


@patch("rasa.tracing.backend_tracing_config._get_tracer_provider")
@patch("rasa.tracing.backend_tracing_config._collect_classes_to_instrument")
@patch("rasa.tracing.backend_tracing_config.instrumentation")
def test_configure_backend_tracing_passes_all_classes(
    mock_instrumentation: MagicMock,
    mock_collect_classes: MagicMock,
    mock_get_tracer_provider: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that configure_backend_tracing passes all collected classes to
    instrumentation."""
    # Setup mocks
    mock_tracer_provider = Mock()
    mock_get_tracer_provider.return_value = mock_tracer_provider

    mock_classes = {
        "command_subclasses": [Mock(), Mock()],
        "policy_subclasses": [Mock()],
        "vector_store_subclasses": [Mock()],
        "custom_action_executor_subclasses": [Mock()],
        "agent_classes": [Mock()],
        "contextual_response_rephraser_class": Mock(),
    }
    mock_collect_classes.return_value = mock_classes

    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: jaeger
                  host: localhost
                  port: 6831
            """
        )
    )

    configure_backend_tracing(str(endpoints_file))

    # Verify all classes were passed to instrumentation
    call_args = mock_instrumentation.instrument.call_args
    assert call_args.kwargs["command_subclasses"] == mock_classes["command_subclasses"]
    assert call_args.kwargs["policy_subclasses"] == mock_classes["policy_subclasses"]
    assert (
        call_args.kwargs["vector_store_subclasses"]
        == mock_classes["vector_store_subclasses"]
    )
    assert (
        call_args.kwargs["custom_action_executor_subclasses"]
        == mock_classes["custom_action_executor_subclasses"]
    )
    assert call_args.kwargs["subagent_classes"] == mock_classes["agent_classes"]
    assert (
        call_args.kwargs["contextual_response_rephraser_class"]
        == mock_classes["contextual_response_rephraser_class"]
    )
