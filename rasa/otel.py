# https://github.com/open-telemetry/opentelemetry-python/issues/1150
import logging
from typing import Text
from rasa.core.utils import EndpointConfig
from opentelemetry import trace, propagators
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
# core exporters
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleExportSpanProcessor,
    BatchExportSpanProcessor,
)
# addl exporters
from opentelemetry.exporter.otlp.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter import jaeger

logger = logging.getLogger(__name__)
logging.getLogger("").handlers = []
logging.basicConfig(format="%(message)s", level=logging.DEBUG)

tracer = None
current_span = None

def init_tracer_endpoint(telemetry):
    # telemetry is the endpoints.yml / telemetry section
    logger.debug(f"init_tracer_endpoint: {telemetry.kwargs}")
    service_name = "rasa"
    if telemetry and telemetry.kwargs.get("service_name"):
        service_name = telemetry.kwargs.get("service_name")

    resource = Resource(attributes={"service.name": service_name})
    trace.set_tracer_provider(TracerProvider(resource=resource))

    # Setup trace exporters
    if telemetry and telemetry.kwargs.get("exporters"):
        trace_exporters = telemetry.kwargs.get("exporters")
        try:
            for t in trace_exporters:
                if t == 'console':
                    logger.debug(f"Starting Console Exporter")
                    trace.get_tracer_provider().add_span_processor(
                        SimpleExportSpanProcessor(ConsoleSpanExporter())
                    )
                if t == 'otlphttp':
                    if trace_exporters["otlphttp"]["endpoint"]:
                        #otlp_exporter = OTLPSpanExporter(endpoint="localhost:55680", insecure=True)
                        logger.debug(f"Starting OTLP Exporter: {trace_exporters['otlphttp']['endpoint']}")
                        insecure = trace_exporters["otlphttp"]["insecure"] if trace_exporters["otlphttp"]["insecure"] else "True"
                        otlp_exporter = OTLPSpanExporter(endpoint=trace_exporters["otlphttp"]["endpoint"], insecure=insecure)
                        trace.get_tracer_provider().add_span_processor(
                            BatchExportSpanProcessor(otlp_exporter)
                        )
                if t == 'jaeger':
                    if trace_exporters["jaeger"]["agent_hosthame"]:
                        agent_port = trace_exporters["jaeger"]["agent_port"] if trace_exporters["jaeger"]["agent_port"] else 6831
                        agent_host_name = trace_exporters["jaeger"]["agent_hosthame"]
                        logger.debug(f"Starting Jaeger Exporter: {agent_host_name}:{agent_port}, service: {service_name}")
                        logger.debug(f"type(agent_port): {type(agent_port)}")
                        jaeger_exporter = jaeger.JaegerSpanExporter(
                            service_name=service_name,
                            agent_host_name=agent_host_name,
                            agent_port=agent_port,
                        )

                        trace.get_tracer_provider().add_span_processor(
                            BatchExportSpanProcessor(jaeger_exporter)
                        )
        except:
            logger.debug(f"error setting up otel exporter: {t}")

    # this call also sets opentracing.tracer
    global tracer
    tracer = trace.get_tracer(__name__)  # returns opentelemetry.sdk.trace.Tracer object
    return tracer

def _show_context(msg, span):
    span_context = span.get_span_context()
    logger.debug(f"{msg}, trace: {span_context.trace_id}, span: {span_context.span_id}")

def start_span(name, attributes=None):
    #span = self.tracer.start_span(name)
    global tracer
    if tracer:
        context = tracer.start_as_current_span(name)
        span = context.args[1]
        _show_context(f"span {name}", span)
        if attributes:
            for k, v in attributes.items():
                if span.is_recording():
                    span.set_attribute(k, v)
                logger.debug(f"set attribute {k} = {v}")
        return context

def start_span_test(name, attributes=None):
    #span = self.tracer.start_span(name)
    global tracer
    global current_span
    if tracer:
        parent_span = current_span
        if name.startswith('channel') or not parent_span:
            current_span = tracer.start_span(name)
        else:
            current_span = tracer.start_span(name, parent=parent_span)
        #_show_context(f"span {name}", current_span)
        with tracer.use_span(current_span, end_on_exit=False):
            if attributes:
                for k, v in attributes.items():
                    if current_span.is_recording():
                        current_span.set_attribute(k, v)
                    logger.debug(f"set attribute {k} = {v}")
        return current_span

def inject():
    # https://github.com/yurishkuro/opentracing-tutorial/blob/7ae271badb867635f6697d9dbe5510c798883ff8/python/lesson03/solution/hello.py#L26
    global tracer
    if tracer:
        #headers = dict()
        #propagators.inject(type(headers).__setitem__, headers)
        headers = {}
        propagators.inject(dict.__setitem__, headers)
        return headers

def extract_start_span(tracer, headers, name, attributes=None):
    span_ctx = tracer.extract(Format.HTTP_HEADERS, headers)
    span_tags = {tags.SPAN_KIND: tags.SPAN_KIND_RPC_SERVER}
    #span = tracer.start_active_span(name)
    logger.debug(f"extract_start_span, child_of: {span_ctx}, span_tags: {span_tags}")
    span = tracer.start_active_span(name, child_of=span_ctx, tags=span_tags)
    #if attributes:
    #    for k, v in attributes.items():
    #        span.span.set_tag(k, v)
    return span
