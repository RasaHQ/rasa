import socket
import threading

import grpc
import pytest
from rasa.utils.endpoints import EndpointConfig

from rasa.tracing import config
from rasa.tracing.config import JaegerTracerConfigurer
from tests.conftest import wait
from tests.tracing import conftest
from tests.tracing.conftest import (
    TRACING_TESTS_FIXTURES_DIRECTORY,
    CapturingTestSpanExporter,
)

UDP_BUFFER_SIZE = 2048


def test_jaeger_config_correctly_extracted() -> None:
    cfg = EndpointConfig(
        host="hostname",
        port=1234,
        username="user",
        password="password",
    )

    extracted = JaegerTracerConfigurer._extract_config(cfg)

    assert extracted["agent_host_name"] == cfg.kwargs["host"]
    assert extracted["agent_port"] == cfg.kwargs["port"]
    assert extracted["username"] == cfg.kwargs["username"]
    assert extracted["password"] == cfg.kwargs["password"]


def test_jaeger_config_sets_defaults() -> None:
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
    endpoints_file = str(TRACING_TESTS_FIXTURES_DIRECTORY / "otlp_endpoints.yml")

    tracer_provider = config.get_tracer_provider(endpoints_file)
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


@pytest.mark.skip(reason="Temporary disabled due to TLS timeout error")
def test_get_tracer_provider_tls_otlp_collector(
    secured_grpc_server: grpc.Server,
    span_exporter: CapturingTestSpanExporter,
    result_available_event: threading.Event,
) -> None:
    endpoints_file = str(TRACING_TESTS_FIXTURES_DIRECTORY / "otlp_endpoints_tls.yml")

    tracer_provider = config.get_tracer_provider(endpoints_file)
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


def test_get_tracer_provider_jaeger(udp_server: socket.socket) -> None:
    endpoints_file = str(TRACING_TESTS_FIXTURES_DIRECTORY / "jaeger_endpoints.yml")

    tracer_provider = config.get_tracer_provider(endpoints_file)
    assert tracer_provider is not None

    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span("jaeger_test_span"):
        pass

    tracer_provider.force_flush()

    message, addr = udp_server.recvfrom(UDP_BUFFER_SIZE)

    batch = conftest.deserialize_jaeger_batch(bytearray(message))

    assert batch.process.serviceName == "rasa"

    assert len(batch.spans) == 1
    assert batch.spans[0].operationName == "jaeger_test_span"
