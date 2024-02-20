from typing import Text, Type

import pytest
from opentelemetry.sdk.trace import TracerProvider

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import (
    MockAgent,
    MockGraphComponent,
    MockGraphTrainer,
    MockLockStore,
    MockMessageProcessor,
    MockTrackerStore,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arg_to_instrument, class_to_instrument",
    [
        ("processor_class", (MockMessageProcessor)),
        ("agent_class", (MockAgent)),
        ("tracker_store_class", (MockTrackerStore)),
        ("graph_node_class", (MockGraphComponent)),
        ("lock_store_class", (MockLockStore)),
        ("graph_trainer_class", (MockGraphTrainer)),
    ],
)
async def test_mark_class_as_instrumented(
    arg_to_instrument: Text,
    class_to_instrument: Type,
    tracer_provider: TracerProvider,
) -> None:
    args = {arg_to_instrument: class_to_instrument}
    instrumentation.instrument(tracer_provider, **args)

    assert instrumentation.class_is_instrumented(class_to_instrument)

    class MockSubClass(class_to_instrument):
        pass

    assert not instrumentation.class_is_instrumented(MockSubClass)
