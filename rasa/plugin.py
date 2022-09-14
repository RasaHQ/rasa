import argparse
import sys
from typing import List, Text, Tuple

import pluggy

from rasa.cli import SubParsersAction

hookspec = pluggy.HookspecMarker("rasa")


@hookspec  # type: ignore[misc]
def refine_cli(
    subparsers: SubParsersAction,
    parent_parsers: List[argparse.ArgumentParser],
) -> None:
    """Customizable hook for adding CLI commands."""


@hookspec  # type: ignore[misc]
def get_version_info() -> Tuple[Text, Text]:
    """Adds extra info on plugin versions."""


@hookspec  # type: ignore[misc]
def get_description() -> Text:
    """Adapts the description to be given to the argument parser."""


@hookspec  # type: ignore[misc]
def get_parent_help_text() -> Text:
    """Adapts the help text of the parent parser."""


@hookspec  # type: ignore[misc]
def get_version_help_text() -> Text:
    """Adapts the help text of the parent parser."""


plugin_manager = pluggy.PluginManager("rasa")
plugin_manager.add_hookspecs(sys.modules["rasa.plugin"])
