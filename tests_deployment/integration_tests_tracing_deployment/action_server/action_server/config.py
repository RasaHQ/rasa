import logging
import os
from typing import Text

from opentelemetry import trace

from rasa.tracing.config import get_tracer_provider

logger = logging.getLogger(__name__)

TRACING_SERVICE_NAME = os.environ.get("TRACING_SERVICE_NAME", "action_server")

os.environ["TRACING_SERVICE_NAME"] = TRACING_SERVICE_NAME


def configure_tracing(endpoints_file: Text) -> None:
    trace.set_tracer_provider(get_tracer_provider(endpoints_file))
