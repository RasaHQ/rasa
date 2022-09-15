import argparse
import functools
import sys
from importlib.metadata import entry_points
from typing import List

import pluggy

from rasa.cli import SubParsersAction

hookspec = pluggy.HookspecMarker("rasa")


@hookspec  # type: ignore[misc]
def refine_cli(
    subparsers: SubParsersAction,
    parent_parsers: List[argparse.ArgumentParser],
) -> None:
    """Customizable hook for adding CLI commands."""


@functools.lru_cache
def plugin_manager() -> pluggy.PluginManager:
    """Initialises a plugin manager which registers hook implementations."""
    _plugin_manager = pluggy.PluginManager("rasa")
    _plugin_manager.add_hookspecs(sys.modules["rasa.plugin"])
    _discover_plugins(_plugin_manager)

    return _plugin_manager


def _discover_plugins(manager: pluggy.PluginManager) -> None:
    rasa_plugin_eps = entry_points().get("rasa.plugin_init", [])
    for plugin in rasa_plugin_eps:
        plugin_init = plugin.load()
        plugin_init(manager)
