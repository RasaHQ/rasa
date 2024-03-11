import collections
import typing
import uuid

import grpc
import pytest
import requests
from google.protobuf.timestamp_pb2 import Timestamp

if typing.TYPE_CHECKING:
    from api_v3.query_service_pb2_grpc import QueryServiceStub

RASA_JAEGER_TRACING_SERVICE_NAME = "rasa-jaeger-testing"
RASA_OTLP_TRACING_SERVICE_NAME = "rasa-otlp-testing"
ACTION_SERVER_JAEGER_TRACING_SERVICE_NAME = "action-server-jaeger-testing"
ACTION_SERVER_OTLP_ACTION_SERVER_NAME = "action-server-otlp-testing"

RASA_SERVER_JAEGER = "http://localhost:5006"
RASA_SERVER_OTLP = "http://localhost:5007"
RASA_SERVER_JAEGER_NO_ACTION_SERVER = "http://localhost:5008"
RASA_SERVER_OTLP_NO_ACTION_SERVER = "http://localhost:5009"


ACTION_SERVER_PARENT_SPAN_NAME = "MessageProcessor._run_action"
ACTION_SERVER_SPAN_NAME = "action_server.run.run_action"
ACTION_SERVER_TRIGGER_MESSAGE = "/goodbye"
ACTION_SERVER_ACTION_TRIGGERED = "action_goodbye"

TraceQueryTimestamps = collections.namedtuple(
    "TraceQueryTimestamps", "min_time max_time"
)

RASA_SERVER_PARENT_SPAN_NAME = "Agent.handle_message"
RASA_SERVER_PROCESSOR_SPAN_NAME = "MessageProcessor.handle_message"
RASA_SERVER_PROCESSOR_SUB_SPAN_NAME = "MessageProcessor.log_message"
RASA_SERVER_TRIGGER_MESSAGE = "/greet"


@pytest.fixture
def jaeger_query_service() -> "QueryServiceStub":
    # needs to be scoped as it can not be resolved outside of integration
    # env. test discovery fails otherwise.
    from api_v3.query_service_pb2_grpc import QueryServiceStub

    channel = grpc.insecure_channel("localhost:16685")
    return QueryServiceStub(channel)


@pytest.fixture
def trace_query_timestamps() -> TraceQueryTimestamps:
    min_time = Timestamp()
    min_time.GetCurrentTime()
    max_time = Timestamp(seconds=min_time.seconds + 5)
    return TraceQueryTimestamps(min_time, max_time)


def send_message_to_rasa_server(server_location: str, message: str = "message") -> str:
    """Send a message to the REST channel and return the sender id."""
    sender_id = str(uuid.uuid4())
    requests.post(
        f"{server_location}/webhooks/rest/webhook",
        json={"sender": sender_id, "message": message},
    )
    return sender_id
