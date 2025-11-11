from __future__ import annotations

import os
from typing import Optional

import grpc
import structlog
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from rasa.tracing.constants import (
    ENDPOINTS_ENDPOINT_KEY,
    ENDPOINTS_INSECURE_KEY,
    ENDPOINTS_METRICS_KEY,
    ENDPOINTS_OTLP_BACKEND_TYPE,
    ENDPOINTS_ROOT_CERTIFICATES_KEY,
    ENDPOINTS_TRACING_SERVICE_NAME_KEY,
)
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

TRACING_SERVICE_NAME = os.environ.get("TRACING_SERVICE_NAME", "rasa")

structlogger = structlog.get_logger()


def configure_metrics(endpoints_file: str) -> None:
    """Configure metrics export for OTLP Collector."""
    metrics_config = read_endpoint_config(endpoints_file, ENDPOINTS_METRICS_KEY)

    if not metrics_config:
        structlogger.debug(
            "metrics_configuration.no_metrics_config",
            event_info=(
                "The OTLP Collector has not been configured to collect "
                "metrics. Skipping."
            ),
        )
        return None

    if metrics_config.type != ENDPOINTS_OTLP_BACKEND_TYPE:
        structlogger.warning(
            "metrics_configuration.unknown_metrics_type",
            event_info=(
                f"Unknown metrics backend type '{metrics_config.type}' "
                f"read from '{endpoints_file}', ignoring."
            ),
        )
        return None

    otlp_exporter = OTLPMetricConfigurer.configure_from_endpoint_config(metrics_config)
    metric_reader = PeriodicExportingMetricReader(otlp_exporter)
    meter_provider = _create_meter_provider(metrics_config, metric_reader)
    set_meter_provider(meter_provider)

    MetricInstrumentProvider().register_instruments()


def _create_meter_provider(
    config: EndpointConfig, metric_reader: PeriodicExportingMetricReader
) -> MeterProvider:
    """Create MeterProvider with service name from configuration."""
    service_name = config.kwargs.get(
        ENDPOINTS_TRACING_SERVICE_NAME_KEY, TRACING_SERVICE_NAME
    )
    return MeterProvider(
        metric_readers=[metric_reader],
        resource=Resource.create({SERVICE_NAME: service_name}),
    )


class OTLPMetricConfigurer:
    """The metric configurer for the OTLP Collector backend."""

    @classmethod
    def configure_from_endpoint_config(
        cls, config: EndpointConfig
    ) -> Optional[OTLPMetricExporter]:
        """Configure metrics for OTLP Collector."""
        insecure = config.kwargs.get(ENDPOINTS_INSECURE_KEY)
        credentials = _get_credentials(config, insecure)

        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=config.kwargs[ENDPOINTS_ENDPOINT_KEY],
            insecure=insecure,
            credentials=credentials,
        )
        structlogger.info(
            "metrics_configuration.otlp_collector_config_configured",
            event_info=(
                f"Registered '{config.type}' endpoint for metrics. "
                f"Metrics will be exported to {config.kwargs[ENDPOINTS_ENDPOINT_KEY]}."
            ),
        )

        return otlp_metric_exporter


def _get_credentials(
    config: EndpointConfig, insecure: Optional[bool]
) -> Optional[grpc.ChannelCredentials]:
    """Get gRPC credentials from configuration."""
    credentials = None
    if not insecure and ENDPOINTS_ROOT_CERTIFICATES_KEY in config.kwargs:
        with open(config.kwargs.get(ENDPOINTS_ROOT_CERTIFICATES_KEY), "rb") as f:
            root_cert = f.read()
        credentials = grpc.ssl_channel_credentials(root_certificates=root_cert)
    return credentials
