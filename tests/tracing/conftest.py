import pathlib
import socket
import threading
from concurrent import futures
from typing import Callable, Generator, Optional, Text

import grpc
import opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc as trace_service
import pytest
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from opentelemetry.exporter.jaeger.thrift.gen.agent.Agent import emitBatch_args
from opentelemetry.exporter.jaeger.thrift.gen.jaeger.ttypes import Batch
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans
from pytest import MonkeyPatch
from rasa.engine.caching import LocalTrainingCache
from thrift.protocol.TCompactProtocol import TCompactProtocol
from thrift.transport.TTransport import TMemoryBuffer

TRACING_TESTS_FIXTURES_DIRECTORY = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def udp_server() -> Generator[socket.socket, None, None]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("localhost", 6832))
    yield sock
    sock.close()


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

    server.add_insecure_port("[::]:4317")

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
        "[::]:4318",
        grpc.ssl_server_credentials(
            private_key_certificate_chain_pairs=[(cert_key, cert)],
        ),
    )
    server.start()
    yield server
    server.stop(None)


def deserialize_jaeger_batch(data: bytearray) -> Batch:
    trans = TMemoryBuffer(data)
    prot = TCompactProtocol(trans)
    prot.readMessageBegin()
    emitBatch = emitBatch_args()  # type: ignore
    emitBatch.read(prot)  # type: ignore
    prot.readMessageEnd()

    return emitBatch.batch


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
