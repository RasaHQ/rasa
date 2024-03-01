from typing import Any

import pytest
from opentelemetry.metrics import Instrument, Histogram
from opentelemetry.sdk.metrics._internal.export import PeriodicExportingMetricReader

from rasa.tracing.constants import (
    LLM_COMMAND_GENERATOR_CPU_USAGE_INSTRUMENT_NAME,
    LLM_COMMAND_GENERATOR_MEMORY_USAGE_INSTRUMENT_NAME,
    LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_INSTRUMENT_NAME,
    LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    INTENTLESS_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
    RASA_CLIENT_REQUEST_DURATION_INSTRUMENT_NAME,
    RASA_CLIENT_REQUEST_BODY_SIZE_INSTRUMENT_NAME,
)
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from tests.tracing.conftest import set_up_test_meter_provider


def test_metric_instrument_provider_is_singleton() -> None:
    instrument_provider_1 = MetricInstrumentProvider()
    instrument_provider_2 = MetricInstrumentProvider()

    assert instrument_provider_1 is instrument_provider_2
    assert instrument_provider_1.instruments is instrument_provider_2.instruments


def test_metric_instrument_provider_register_instruments(
    periodic_exporting_metric_reader: PeriodicExportingMetricReader,
) -> None:
    set_up_test_meter_provider(periodic_exporting_metric_reader)

    instrument_provider = MetricInstrumentProvider()
    assert instrument_provider.instruments == {}

    instrument_provider.register_instruments()
    assert len(instrument_provider.instruments) > 0
    assert all(
        [
            isinstance(instrument, Instrument)
            for instrument in instrument_provider.instruments.values()
        ]
    )


@pytest.mark.parametrize(
    "instrument_name, expected_instrument_type",
    [
        (LLM_COMMAND_GENERATOR_CPU_USAGE_INSTRUMENT_NAME, Histogram),
        (LLM_COMMAND_GENERATOR_MEMORY_USAGE_INSTRUMENT_NAME, Histogram),
        (LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_INSTRUMENT_NAME, Histogram),
        (LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_INSTRUMENT_NAME, Histogram),
        (ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME, Histogram),
        (INTENTLESS_POLICY_LLM_RESPONSE_DURATION_INSTRUMENT_NAME, Histogram),
        (
            CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_INSTRUMENT_NAME,
            Histogram,
        ),
        (RASA_CLIENT_REQUEST_DURATION_INSTRUMENT_NAME, Histogram),
        (RASA_CLIENT_REQUEST_BODY_SIZE_INSTRUMENT_NAME, Histogram),
    ],
)
def test_metric_instrument_provider_get_instrument(
    instrument_name: str,
    expected_instrument_type: Any,
    periodic_exporting_metric_reader: PeriodicExportingMetricReader,
) -> None:
    # arrange
    set_up_test_meter_provider(periodic_exporting_metric_reader)
    instrument_provider = MetricInstrumentProvider()
    instrument_provider.register_instruments()

    # act
    instrument = instrument_provider.get_instrument(instrument_name)

    # assert
    assert instrument is not None
    assert isinstance(instrument, expected_instrument_type)
