import textwrap
from pathlib import Path

import structlog
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from pytest import LogCaptureFixture

from rasa.tracing.constants import ENDPOINTS_METRICS_KEY
from rasa.tracing.metrics_config import (
    OTLPMetricConfigurer,
    configure_metrics,
)
from rasa.utils.endpoints import read_endpoint_config
from tests.tracing.conftest import TRACING_TESTS_FIXTURES_DIRECTORY
from tests.utilities import filter_logs


def test_configure_otlp_metric_exporter() -> None:
    """Test that OTLP metric exporter is correctly configured."""
    endpoints_file = str(
        TRACING_TESTS_FIXTURES_DIRECTORY / "metrics_otlp_endpoints.yml"
    )
    metrics_config = read_endpoint_config(endpoints_file, ENDPOINTS_METRICS_KEY)
    assert metrics_config is not None

    otlp_metric_exporter = OTLPMetricConfigurer.configure_from_endpoint_config(
        metrics_config
    )
    assert isinstance(otlp_metric_exporter, OTLPMetricExporter)


def test_log_warning_with_non_otlp_backend(tmp_path: Path) -> None:
    """Test that warning is logged for non-OTLP backend."""
    test_metrics_type = "unsupported"
    endpoints_file = tmp_path / "metrics_endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            f"""
            metrics:
                type: {test_metrics_type}
            """
        )
    )

    with structlog.testing.capture_logs() as caplog:
        configure_metrics(str(endpoints_file))
        logs = filter_logs(caplog, "metrics_configuration.unknown_metrics_type")

        assert len(logs) == 1
        assert (
            f"Unknown metrics backend type '{test_metrics_type}' "
            f"read from '{endpoints_file!s}', ignoring."
        ) in logs[0]["event_info"]


def test_log_debug_with_no_metrics_configured(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Test that debug message is logged when no metrics configured."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            action_endpoint:
                url: "http://localhost:5056/webhook"
            """
        )
    )

    with structlog.testing.capture_logs() as caplog:
        configure_metrics(str(endpoints_file))
        logs = filter_logs(caplog, "metrics_configuration.no_metrics_config")

        assert len(logs) == 1
        assert (
            "The OTLP Collector has not been configured to collect metrics. Skipping."
        ) in logs[0]["event_info"]
