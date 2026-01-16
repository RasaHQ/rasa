import pathlib
import threading
from concurrent import futures
from typing import Callable, Generator, Optional, Text

import grpc
import opentelemetry.metrics
import opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc as trace_service
import pytest
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from pytest import MonkeyPatch

from rasa.engine.caching import LocalTrainingCache
from rasa.tracing.constants import (
    LANGFUSE_ENV_VAR_DEBUG,
    LANGFUSE_ENV_VAR_MEDIA_UPLOAD_THREAD_COUNT,
    LANGFUSE_ENV_VAR_OTEL_HOST,
    LANGFUSE_ENV_VAR_PUBLIC_KEY,
    LANGFUSE_ENV_VAR_RELEASE,
    LANGFUSE_ENV_VAR_SAMPLE_RATE,
    LANGFUSE_ENV_VAR_SECRET_KEY,
    LANGFUSE_ENV_VAR_TIMEOUT,
    LANGFUSE_ENV_VAR_TRACING_ENVIRONMENT,
)

TRACING_TESTS_FIXTURES_DIRECTORY = pathlib.Path(__file__).parent / "fixtures"


class CapturingTestSpanExporter(trace_service.TraceServiceServicer):
    def __init__(self) -> None:
        self.spans: Optional[RepeatedCompositeFieldContainer[ResourceSpans]] = None

    def Export(
        self, request: ExportTraceServiceRequest, context: grpc.ServicerContext
    ) -> ExportTraceServiceResponse:
        self.spans = request.resource_spans

        return ExportTraceServiceResponse()


@pytest.fixture
def span_exporter() -> CapturingTestSpanExporter:
    return CapturingTestSpanExporter()


@pytest.fixture
def grpc_server(
    span_exporter: CapturingTestSpanExporter,
) -> Generator[grpc.Server, None, None]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    trace_service.add_TraceServiceServicer_to_server(  # type: ignore
        span_exporter, server
    )

    # Use a different port to avoid conflicts
    server.add_insecure_port("[::]:4319")

    server.start()
    yield server
    server.stop(None)


@pytest.fixture
def secured_grpc_server(
    span_exporter: CapturingTestSpanExporter,
) -> Generator[grpc.Server, None, None]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    trace_service.add_TraceServiceServicer_to_server(  # type: ignore
        span_exporter, server
    )

    with open(TRACING_TESTS_FIXTURES_DIRECTORY / "cert.pem", "rb") as f:
        cert = f.read()
    with open(TRACING_TESTS_FIXTURES_DIRECTORY / "cert-key.pem", "rb") as f:
        cert_key = f.read()

    server.add_secure_port(
        "[::]:4321",
        grpc.ssl_server_credentials(
            private_key_certificate_chain_pairs=[(cert_key, cert)],
        ),
    )
    server.start()
    yield server
    server.stop(None)


@pytest.fixture()
def config_path() -> Text:
    return str(TRACING_TESTS_FIXTURES_DIRECTORY / "short_config.yml")


@pytest.fixture()
def domain_path() -> Text:
    return str(TRACING_TESTS_FIXTURES_DIRECTORY / "default_domain.yml")


@pytest.fixture
def data_path() -> Text:
    return str(TRACING_TESTS_FIXTURES_DIRECTORY / "data")


@pytest.fixture()
def local_cache_creator(monkeypatch: MonkeyPatch) -> Callable[..., LocalTrainingCache]:
    def create_local_cache(path: pathlib.Path) -> LocalTrainingCache:
        monkeypatch.setattr(LocalTrainingCache, "_get_cache_location", lambda: path)
        return LocalTrainingCache()

    return create_local_cache


@pytest.fixture()
def temp_cache(
    tmp_path: pathlib.Path, local_cache_creator: Callable
) -> LocalTrainingCache:
    return local_cache_creator(tmp_path)


@pytest.fixture
def result_available_event() -> threading.Event:
    return threading.Event()


@pytest.fixture
def periodic_exporting_metric_reader() -> PeriodicExportingMetricReader:
    return PeriodicExportingMetricReader(ConsoleMetricExporter())


def set_up_test_meter_provider(
    metric_reader: MetricReader,
) -> Generator[MeterProvider, None, None]:
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    opentelemetry.metrics.set_meter_provider(meter_provider)
    yield meter_provider
    meter_provider.shutdown()


# Cleanup any LANGFUSE_* env vars after each test to prevent leakage across tests
_LANGFUSE_ENV_VARS = [
    LANGFUSE_ENV_VAR_PUBLIC_KEY,
    LANGFUSE_ENV_VAR_SECRET_KEY,
    LANGFUSE_ENV_VAR_OTEL_HOST,
    LANGFUSE_ENV_VAR_TIMEOUT,
    LANGFUSE_ENV_VAR_DEBUG,
    LANGFUSE_ENV_VAR_TRACING_ENVIRONMENT,
    LANGFUSE_ENV_VAR_RELEASE,
    LANGFUSE_ENV_VAR_MEDIA_UPLOAD_THREAD_COUNT,
    LANGFUSE_ENV_VAR_SAMPLE_RATE,
]


@pytest.fixture(autouse=True)
def _cleanup_langfuse_env(monkeypatch):
    yield
    for var in _LANGFUSE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
