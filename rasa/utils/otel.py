# https://github.com/open-telemetry/opentelemetry-python/issues/1150
import logging
from typing import Text
from rasa.core.utils import EndpointConfig
# Jaeger
from jaeger_client import Config
from opentracing import Format, UnsupportedFormatException
from opentracing.ext import tags

logger = logging.getLogger(__name__)
logging.getLogger("").handlers = []
logging.basicConfig(format="%(message)s", level=logging.DEBUG)

tracer = None

def init_tracer_endpoint(endpoints):
    service_name = "rasa"
    if endpoints and endpoints.telemetry and endpoints.telemetry.kwargs.get("service_name"):
        service_name = endpoints.telemetry.kwargs.get("service_name")
    logger.debug(f"initializing tracing, service_name: {service_name}")

    config = Config(
        config={ # usually read from some yaml config
            'sampler': {
                'type': 'const',
                'param': 1,
            },
            'logging': True,
            'reporter_batch_size': 1,
        },
        service_name=service_name,
    )

    # this call also sets opentracing.tracer
    global tracer
    tracer = config.initialize_tracer()
    return tracer


def start_span(name, attributes=None):
    #span = self.tracer.start_span(name)
    global tracer
    if tracer:
        span = tracer.start_active_span(name)
        if attributes:
            for k, v in attributes.items():
                span.span.set_tag(k, v)
        return span

def inject(url):
    # https://github.com/yurishkuro/opentracing-tutorial/blob/7ae271badb867635f6697d9dbe5510c798883ff8/python/lesson03/solution/hello.py#L26
    global tracer
    if tracer:
        span = tracer.active_span
        logger.debug(f"otel injecting headers, span: {span}")
        span.set_tag(tags.HTTP_METHOD, 'POST')
        span.set_tag(tags.HTTP_URL, url)
        span.set_tag(tags.SPAN_KIND, tags.SPAN_KIND_RPC_CLIENT)
        headers = {}
        tracer.inject(span, Format.HTTP_HEADERS, headers)
        #span_header = self.tracer.inject(span, Format.HTTP_HEADERS, headers)
        # tracer.inject(child_span.context, 'zipkin-span-format', text_carrier)
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
