import collections
import typing

import grpc
import pytest
from google.protobuf.timestamp_pb2 import Timestamp

if typing.TYPE_CHECKING:
    from api_v3.query_service_pb2_grpc import QueryServiceStub

RASA_JAEGER_TRACING_SERVICE_NAME = "rasa-jaeger-testing"
RASA_OTLP_TRACING_SERVICE_NAME = "rasa-otlp-testing"
ACTION_SERVER_JAEGER_TRACING_SERVICE_NAME = "rasa_sdk"
ACTION_SERVER_OTLP_ACTION_SERVER_NAME = "rasa_sdk"

RASA_SERVER_JAEGER = "http://localhost:5006"
RASA_SERVER_OTLP = "http://localhost:5007"
RASA_SERVER_JAEGER_NO_ACTION_SERVER = "http://localhost:5008"
RASA_SERVER_OTLP_NO_ACTION_SERVER = "http://localhost:5009"


ACTION_SERVER_PARENT_SPAN_NAME = "MessageProcessor._run_action"
ACTION_SERVER_PARENT_SUB_SPAN_NAME = "HTTPCustomActionExecutor.run"
ACTION_SERVER_SPAN_NAME = "ActionExecutor.run"
ACTION_SERVER_TRIGGER_MESSAGE = "/goodbye"
ACTION_SERVER_ACTION_TRIGGERED = "action_goodbye"
ACTION_SERVER_FORM_TRIGGER_MESSAGE = "/request_name"
ACTION_SERVER_FORM_FILL_MESSAGE = "Tom"
ACTION_SERVER_FORM_VALIDATION_ACTION_TRIGGERED = "validate_name_form"


TraceQueryTimestamps = collections.namedtuple(
    "TraceQueryTimestamps", "min_time max_time"
)

RASA_SERVER_PARENT_SPAN_NAME = "Agent.handle_message"
RASA_SERVER_PROCESSOR_SPAN_NAME = "MessageProcessor.handle_message"
RASA_SERVER_PROCESSOR_SUB_SPAN_NAME = "MessageProcessor.log_message"
DIRECT_CUSTOM_ACTION_EXECUTION_SUB_SPAN_NAME = "DirectCustomActionExecutor.run"
RASA_SERVER_TRIGGER_MESSAGE = "/goodbye"


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
