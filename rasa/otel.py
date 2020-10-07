# https://github.com/open-telemetry/opentelemetry-python/issues/1150
import logging
from typing import Text
from opentelemetry import trace
from opentelemetry.exporter import jaeger
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchExportSpanProcessor
#from jaeger_client import Config
#import otel

logger = logging.getLogger(__name__)
logging.getLogger("").handlers = []
logging.basicConfig(format="%(message)s", level=logging.DEBUG)

class Tracer(object):
    def __init__(self, service_name: Text = "rasa", agent_host_name: Text="localhost", agent_port: int=6831):
        logger.debug("Tracer - init")
        trace.set_tracer_provider(TracerProvider())

        jaeger_exporter = jaeger.JaegerSpanExporter(
            service_name=service_name,
            agent_host_name=agent_host_name,
            agent_port=agent_port,
        )

        trace.get_tracer_provider().add_span_processor(
            BatchExportSpanProcessor(jaeger_exporter)
        )

        #self.tracer = trace.get_tracer(__name__)
        self.tracer = trace.get_tracer(__name__)
        logger.debug(f"Tracer - exit init, name: {__name__}")

    def start_span(self, name, attributes=None):
        span = self.tracer.start_as_current_span(name, attributes=attributes)
        return span

tracer = Tracer("rasa", "localhost", 6831)
