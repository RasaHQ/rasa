import argparse
import logging
from typing import Optional, TYPE_CHECKING, List, Text, Union

import pluggy
from rasa.cli import SubParsersAction

from rasa.cli import x as rasa_x
from rasa.utils.endpoints import EndpointConfig

from rasa.core.auth_retry_tracker_store import AuthRetryTrackerStore
from rasa.core.secrets_manager.factory import load_secret_manager

from rasa.tracing import config

if TYPE_CHECKING:
    from rasa.core.brokers.broker import EventBroker
    from rasa.core.tracker_store import TrackerStore
    from rasa.shared.core.domain import Domain
    from rasa.anonymization.anonymization_pipeline import AnonymizationPipeline

hookimpl = pluggy.HookimplMarker("rasa")
logger = logging.getLogger(__name__)


@hookimpl  # type: ignore[misc]
def refine_cli(
    subparsers: SubParsersAction,
    parent_parsers: List[argparse.ArgumentParser],
) -> None:
    from rasa.cli import e2e_test, inspect, markers
    from rasa.cli.studio import studio

    from rasa.cli import license as license_cli

    e2e_test.add_subparser(subparsers, parent_parsers)
    studio.add_subparser(subparsers, parent_parsers)
    license_cli.add_subparser(subparsers, parent_parsers)
    markers.add_subparser(subparsers, parent_parsers)
    inspect.add_subparser(subparsers, parent_parsers)
    return None


@hookimpl  # type: ignore[misc]
def configure_commandline(cmdline_arguments: argparse.Namespace) -> Optional[Text]:
    endpoints_file = None

    if cmdline_arguments.func.__name__ == "rasa_x":
        _, endpoints_file = rasa_x._get_credentials_and_endpoints_paths(
            cmdline_arguments
        )
    elif "endpoints" in cmdline_arguments:
        endpoints_file = cmdline_arguments.endpoints

    if endpoints_file is not None:
        tracer_provider = config.get_tracer_provider(endpoints_file)
        config.configure_tracing(tracer_provider)
        config.configure_metrics(endpoints_file)

    return endpoints_file


@hookimpl  # type: ignore[misc]
def init_telemetry(endpoints_file: Optional[Text]) -> None:
    import rasa.telemetry

    rasa.telemetry.identify_endpoint_config_traits(endpoints_file)


@hookimpl  # type: ignore[misc]
def init_managers(endpoints_file: Optional[Text]) -> None:
    load_secret_manager(endpoints_file)


@hookimpl  # type: ignore[misc]
def create_tracker_store(
    endpoint_config: Union["TrackerStore", "EndpointConfig"],
    domain: "Domain",
    event_broker: Optional["EventBroker"],
) -> "TrackerStore":
    if isinstance(endpoint_config, EndpointConfig):
        return AuthRetryTrackerStore(
            endpoint_config=endpoint_config, domain=domain, event_broker=event_broker
        )
    return endpoint_config


@hookimpl  # type: ignore[misc]
def init_anonymization_pipeline(endpoints_file: Optional[Text]) -> None:
    """Hook implementation for initializing the anonymization pipeline."""
    from rasa.anonymization.anonymization_pipeline import load_anonymization_pipeline

    load_anonymization_pipeline(endpoints_file)


@hookimpl  # type: ignore[misc]
def get_anonymization_pipeline() -> Optional["AnonymizationPipeline"]:
    """Hook implementation for getting the anonymization pipeline."""
    from rasa.anonymization.anonymization_pipeline import AnonymizationPipelineProvider

    return AnonymizationPipelineProvider().get_anonymization_pipeline()


@hookimpl  # type: ignore[misc]
def after_server_stop() -> None:
    """Hook implementation for stopping the anonymization pipeline."""
    from rasa.anonymization.anonymization_pipeline import AnonymizationPipelineProvider

    anon_pipeline = AnonymizationPipelineProvider().get_anonymization_pipeline()

    if anon_pipeline is not None:
        anon_pipeline.stop()
