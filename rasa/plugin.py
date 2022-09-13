import argparse
from typing import List

import pluggy


hookspec = pluggy.HookspecMarker("rasa")
hookimpl = pluggy.HookimplMarker("rasa")


class CLICommandSpec:
    """A hook specification namespace."""

    @hookspec
    def refine_cli(self, arg: argparse._SubParsersAction, parent_parsers: List[argparse.ArgumentParser]) -> None:
        """Customizable hook for adding CLI commands."""


plugin_manager = pluggy.PluginManager("rasa")
plugin_manager.add_hookspecs(CLICommandSpec)
