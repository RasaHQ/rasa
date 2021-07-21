import logging
import typing
from typing import Optional
from os import linesep

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


class ConsoleCompactSpanExporter(SpanExporter):
    """Implementation of :class:`SpanExporter` that prints spans to the
    console.

    This class can be used for diagnostic purposes. It prints the exported
    spans to the console STDOUT.
    """

    def __init__(
        self,
        service_name: Optional[str] = None,
        formatter: typing.Callable[
            [ReadableSpan], str
        ] = lambda span: span.to_json(indent=None)
        + linesep,
    ):
        self.formatter = formatter
        self.service_name = service_name

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            logger.debug(self.formatter(span))
        return SpanExportResult.SUCCESS
