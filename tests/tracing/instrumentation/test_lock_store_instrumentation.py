from typing import Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import MockLockStore


@pytest.mark.asyncio
async def test_tracing_for_lock(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        lock_store_class=MockLockStore,
    )

    lock_store = MockLockStore()

    async with lock_store.lock("conversation_id"):
        pass

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockLockStore.lock"
    assert captured_span.attributes == {
        "lock_store_class": MockLockStore.__name__,
    }
