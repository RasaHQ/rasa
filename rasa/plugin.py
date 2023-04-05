import argparse
import functools
import sys
from typing import List, Optional, TYPE_CHECKING, Text, Tuple, Union

import pluggy

from rasa.cli import SubParsersAction

if TYPE_CHECKING:
    from rasa.core.brokers.broker import EventBroker
    from rasa.core.tracker_store import TrackerStore
    from rasa.shared.core.domain import Domain
    from rasa.utils.endpoints import EndpointConfig


hookspec = pluggy.HookspecMarker("rasa")


@functools.lru_cache(maxsize=2)
def plugin_manager() -> pluggy.PluginManager:
    """Initialises a plugin manager which registers hook implementations."""
    _plugin_manager = pluggy.PluginManager("rasa")
    _plugin_manager.add_hookspecs(sys.modules["rasa.plugin"])
    _discover_plugins(_plugin_manager)

    return _plugin_manager


def _discover_plugins(manager: pluggy.PluginManager) -> None:
    try:
        # rasa_plus is an enterprise-ready version of rasa open source
        # which extends existing functionality via plugins
        import rasa_plus

        rasa_plus.init_hooks(manager)
    except ModuleNotFoundError:
        pass


@hookspec  # type: ignore[misc]
def refine_cli(
    subparsers: SubParsersAction,
    parent_parsers: List[argparse.ArgumentParser],
) -> None:
    """Customizable hook for adding CLI commands."""


@hookspec  # type: ignore[misc, empty-body]
def get_version_info() -> Tuple[Text, Text]:
    """Hook specification for getting plugin version info."""


@hookspec  # type: ignore[misc]
def configure_commandline(cmdline_arguments: argparse.Namespace) -> Optional[Text]:
    """Hook specification for configuring plugin CLI."""


@hookspec  # type: ignore[misc]
def init_telemetry(endpoints_file: Optional[Text]) -> None:
    """Hook specification for initialising plugin telemetry."""


@hookspec  # type: ignore[misc]
def init_managers(endpoints_file: Optional[Text]) -> None:
    """Hook specification for initialising managers."""


@hookspec(firstresult=True)  # type: ignore[misc]
def create_tracker_store(  # type: ignore[empty-body]
    endpoint_config: Union["TrackerStore", "EndpointConfig"],
    domain: "Domain",
    event_broker: Optional["EventBroker"],
) -> "TrackerStore":
    """Hook specification for wrapping with AuthRetryTrackerStore."""
