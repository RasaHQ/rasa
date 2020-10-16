# https://github.com/open-telemetry/opentelemetry-python/issues/1150
import logging
from typing import Text
from rasa.core.utils import EndpointConfig
# OpenTelemetry
from opentelemetry import trace, propagators
from opentelemetry.exporter import jaeger
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchExportSpanProcessor
# Jaeger
from jaeger_client import Config
from opentracing import Format, UnsupportedFormatException
from opentracing.ext import tags

logger = logging.getLogger(__name__)
logging.getLogger("").handlers = []
logging.basicConfig(format="%(message)s", level=logging.DEBUG)

class Tracer(object):
    def __init__(self, type: Text = None, service_name: Text = "rasa", agent_host_name: Text="localhost", agent_port: int=6831):
        self.type = type
        if self.type == 'open_telemetry':
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
        elif self.type == 'jaeger':
            config = Config(
                config={ # usually read from some yaml config
                    'sampler': {
                        'type': 'const',
                        'param': 1,
                    },
                    'logging': True,
                },
                service_name=service_name,
                validate=True,
            )

            # this call also sets opentracing.tracer
            self.tracer = config.initialize_tracer()

    def start_span(self, name, attributes=None):
        if self.type == 'open_telemetry':
            span = self.tracer.start_as_current_span(name, attributes=attributes)
            return span
        elif self.type == 'jaeger':
            #span = self.tracer.start_span(name)
            span = self.tracer.start_active_span(name)
            if attributes:
                for k, v in attributes.items():
                    span.span.set_tag(k, v)
            return span

    def inject(self, url):
        if self.type == 'open_telemetry':
            headers = {}
            propagators.inject(type(headers).__setitem__, headers)
            return headers
        elif self.type == 'jaeger':
            # https://github.com/yurishkuro/opentracing-tutorial/blob/7ae271badb867635f6697d9dbe5510c798883ff8/python/lesson03/solution/hello.py#L26
            headers = {}
            span = self.tracer.active_span
            span.set_tag(tags.HTTP_METHOD, 'POST')
            span.set_tag(tags.HTTP_URL, url)
            span.set_tag(tags.SPAN_KIND, tags.SPAN_KIND_RPC_CLIENT)
            headers = self.tracer.inject(span, Format.HTTP_HEADERS, headers)
            # tracer.inject(child_span.context, 'zipkin-span-format', text_carrier)
            return headers

# def handle_request(ctx):
#    extractors = propagation.http_extractors()
#    with propagation.extract(request.headers, context=ctx, extractors=extractors) as ctx1:
#        with tracer.start_as_current_span("service-span", context=ctx1) as ctx2:
#            with tracer.start_as_current_span("external-req-span", context=ctx2) as ctx3:

tracer = {}

def init_tracer(telemetry, service_name=None):
    # parms read from `endpoints.yml`
    if telemetry:
        if service_name:
            return Tracer(telemetry.type, service_name, telemetry.kwargs.get("agent_hostname"), telemetry.kwargs.get("agent_port"))
        else:
            global tracer
            tracer = Tracer(telemetry.type, telemetry.kwargs.get("service_name"), telemetry.kwargs.get("agent_hostname"), telemetry.kwargs.get("agent_port"))
