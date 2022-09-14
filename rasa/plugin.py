import argparse
import sys
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


plugin_manager = pluggy.PluginManager("rasa")
plugin_manager.add_hookspecs(sys.modules["rasa.plugin"])
