import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter
)
from opentelemetry.propagate import inject, extract
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from rasa.opentelemetry.exporter.console_compact import ConsoleCompactSpanExporter

from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)


class Tracer:
    tracer = None

    @classmethod
    def init(cls, config):
        service_name = config.get('service_name')
        exporters = config.get('exporters')
        if not service_name or not exporters:
            return

        logger.info(f"Starting tracing for {service_name}")

        resource = Resource(attributes={"service.name": service_name})
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Setup trace exporters
        if exporters:
            for t in exporters:
                try:
                    if t == 'console':
                        logger.debug(f"Starting Console Exporter")
                        log_format = exporters["console"]["format"]
                        exporter = ConsoleCompactSpanExporter() if log_format == "compact" else ConsoleSpanExporter()
                        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                    if t == 'jaeger':
                        if exporters["jaeger"]["agent_hostname"]:
                            agent_port = exporters["jaeger"]["agent_port"]
                            agent_host_name = exporters["jaeger"]["agent_hostname"]
                            logger.debug(f"Starting Jaeger Exporter: {agent_host_name}:{agent_port}")
                            jaeger_exporter = JaegerExporter(agent_host_name=agent_host_name, agent_port=agent_port)
                            tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
                except Exception as e:
                    import traceback
                    logger.error(f"Exception setting up telemetry exporter {t} : " + type(e).__name__ + " " + str(e))
                    traceback.print_exception(type(e), e, e.__traceback__)

        cls.tracer = trace.get_tracer(__name__)

        LoggingInstrumentor().instrument()

    @classmethod
    def start_span(cls, name, attributes=None, context=None) -> trace.Span:
        if not cls.tracer:
            trace.Span()

        logger.info(f"Start span: {name}")
        span = cls.tracer.start_as_current_span(name, attributes=attributes, context=context)
        return span

    @classmethod
    def inject(cls):
        headers = {}
        inject(carrier=headers)
        return headers

    @classmethod
    def extract(cls, headers):
        context = extract(carrier=headers)
        return context
