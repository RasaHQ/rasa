import argparse
import logging
from typing import TYPE_CHECKING, List, Optional, Text, Union

import pluggy

# IMPORTANT: do not import anything from rasa here - use scoped imports
#  this avoids circular imports, as the hooks are used in different places
#  across the codebase.

if TYPE_CHECKING:
    from rasa.cli import SubParsersAction
    from rasa.core.brokers.broker import EventBroker
    from rasa.core.tracker_stores.tracker_store import TrackerStore
    from rasa.shared.core.domain import Domain
    from rasa.utils.endpoints import EndpointConfig

hookimpl = pluggy.HookimplMarker("rasa")
logger = logging.getLogger(__name__)


@hookimpl
def refine_cli(
    subparsers: "SubParsersAction",
    parent_parsers: List[argparse.ArgumentParser],
) -> None:
    from rasa.cli import dialogue_understanding_test, e2e_test, inspect, markers
    from rasa.cli import license as license_cli
    from rasa.cli.studio import studio

    e2e_test.add_subparser(subparsers, parent_parsers)
    dialogue_understanding_test.add_subparser(subparsers, parent_parsers)
    studio.add_subparser(subparsers, parent_parsers)
    license_cli.add_subparser(subparsers, parent_parsers)
    markers.add_subparser(subparsers, parent_parsers)
    inspect.add_subparser(subparsers, parent_parsers)
    return None


@hookimpl
def configure_commandline(cmdline_arguments: argparse.Namespace) -> Optional[Text]:
    from rasa.cli import x as rasa_x
    from rasa.tracing.backend_tracing_config import configure_backend_tracing
    from rasa.tracing.langfuse_config import configure_langfuse
    from rasa.tracing.metrics_config import configure_metrics

    endpoints_file = None

    if cmdline_arguments.func.__name__ == "rasa_x":
        _, endpoints_file = rasa_x._get_credentials_and_endpoints_paths(
            cmdline_arguments
        )
    elif "endpoints" in cmdline_arguments:
        endpoints_file = cmdline_arguments.endpoints

    if endpoints_file is not None:
        configure_backend_tracing(endpoints_file)
        configure_langfuse(endpoints_file)
        configure_metrics(endpoints_file)

    return endpoints_file


@hookimpl
def init_telemetry(endpoints_file: Optional[Text]) -> None:
    import rasa.telemetry

    rasa.telemetry.identify_endpoint_config_traits(endpoints_file)


@hookimpl
def init_managers(endpoints_file: Optional[Text]) -> None:
    from rasa.core.secrets_manager.factory import load_secret_manager

    load_secret_manager(endpoints_file)


@hookimpl
def create_tracker_store(
    endpoint_config: Union["TrackerStore", "EndpointConfig"],
    domain: "Domain",
    event_broker: Optional["EventBroker"],
) -> "TrackerStore":
    from rasa.core.tracker_stores.auth_retry_tracker_store import AuthRetryTrackerStore
    from rasa.utils.endpoints import EndpointConfig

    if isinstance(endpoint_config, EndpointConfig):
        return AuthRetryTrackerStore(
            endpoint_config=endpoint_config, domain=domain, event_broker=event_broker
        )
    return endpoint_config
