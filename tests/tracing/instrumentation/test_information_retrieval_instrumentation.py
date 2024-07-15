from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import MockInformationRetrieval


async def test_tracing_information_retrieval_search(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockInformationRetrieval

    instrumentation.instrument(
        tracer_provider,
        vector_store_subclasses=[component_class],
    )

    mock_information_retrieval = component_class()
    await mock_information_retrieval.search(
        query="What functionality does Finley support?",
        tracker_state={},
    )
    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockInformationRetrieval.search"

    assert captured_span.attributes == {
        "query": "What functionality does Finley support?",
        "document_metadata": '[{"source": "docs/test.txt"}]',
    }
