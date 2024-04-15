from __future__ import annotations

import argparse
import functools
import sys
from typing import Any, List, Optional, TYPE_CHECKING, Text, Union

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
    init_hooks(_plugin_manager)

    return _plugin_manager


def init_hooks(manager: pluggy.PluginManager) -> None:
    """Initialise hooks into rasa."""
    import rasa.utils.licensing
    from rasa import hooks

    rasa.utils.licensing.validate_license_from_env()

    manager.register(hooks)


@hookspec  # type: ignore[misc]
def refine_cli(
    subparsers: SubParsersAction,
    parent_parsers: List[argparse.ArgumentParser],
) -> None:
    """Customizable hook for adding CLI commands."""


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


@hookspec(firstresult=True)  # type: ignore[misc]
def init_anonymization_pipeline(endpoints_file: Optional[Text]) -> None:
    """Hook specification for initialising the anonymization pipeline."""


@hookspec(firstresult=True)  # type: ignore[misc]
def get_anonymization_pipeline() -> Optional[Any]:
    """Hook specification for getting the anonymization pipeline."""


@hookspec  # type: ignore[misc]
def after_server_stop() -> None:
    """Hook specification for stopping the server.

    Use this hook to de-initialize any resources that require explicit cleanup like,
    thread shutdown, closing connections, etc.
    """
