import argparse
from typing import List

import pluggy


hookspec = pluggy.HookspecMarker("rasa")


class CLICommandSpec:
    """A hook specification namespace."""

    @hookspec  # type: ignore[misc]
    def refine_cli(
        self,
        subparsers: argparse._SubParsersAction,
        parent_parsers: List[argparse.ArgumentParser],
    ) -> None:
        """Customizable hook for adding CLI commands."""


plugin_manager = pluggy.PluginManager("rasa")
plugin_manager.add_hookspecs(CLICommandSpec)
