from __future__ import annotations

import argparse
import functools
import sys
from typing import TYPE_CHECKING, List, Optional, Text, Union

import pluggy

from rasa.cli import SubParsersAction

if TYPE_CHECKING:
    from rasa.core.brokers.broker import EventBroker
    from rasa.core.tracker_stores.tracker_store import TrackerStore
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker
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
    from rasa import hooks

    manager.register(hooks)


@hookspec
def refine_cli(
    subparsers: SubParsersAction,
    parent_parsers: List[argparse.ArgumentParser],
) -> None:
    """Customizable hook for adding CLI commands."""


@hookspec
def configure_commandline(cmdline_arguments: argparse.Namespace) -> Optional[Text]:
    """Hook specification for configuring plugin CLI."""


@hookspec
def init_telemetry(endpoints_file: Optional[Text]) -> None:
    """Hook specification for initialising plugin telemetry."""


@hookspec
def init_managers(endpoints_file: Optional[Text]) -> None:
    """Hook specification for initialising managers."""


@hookspec(firstresult=True)
def create_tracker_store(  # type: ignore[empty-body]
    endpoint_config: Union["TrackerStore", "EndpointConfig"],
    domain: "Domain",
    event_broker: Optional["EventBroker"],
) -> "TrackerStore":
    """Hook specification for wrapping with AuthRetryTrackerStore."""


@hookspec
def after_server_stop() -> None:
    """Hook specification for stopping the server.

    Use this hook to de-initialize any resources that require explicit cleanup like,
    thread shutdown, closing connections, etc.
    """


@hookspec
def after_new_user_message(tracker: "DialogueStateTracker") -> None:
    """Hook specification for after a new user message is received."""


@hookspec
def after_action_executed(tracker: "DialogueStateTracker") -> None:
    """Hook specification for after an action is executed."""
