import argparse
import functools
import sys
from typing import Dict, List, Optional, Text, Tuple
import typing

import pluggy

from rasa.cli import SubParsersAction
from rasa.engine.storage.storage import ModelMetadata
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message

if typing.TYPE_CHECKING:
    from rasa.rasa.engine.graph import SchemaNode

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


@hookspec  # type: ignore[misc]
def modify_default_recipe_graph_train_nodes(
    train_nodes: Dict[Text, "SchemaNode"]
) -> None:
    """Hook specification to modify the default recipe graph for training.

    Modifications are made in-place.
    """


@hookspec  # type: ignore[misc]
def modify_default_recipe_graph_predict_nodes(
    predict_nodes: Dict[Text, "SchemaNode"]
) -> None:
    """Hook specification to modify the default recipe graph for prediction.

    Modifications are made in-place.
    """


@hookspec  # type: ignore[misc]
def get_version_info() -> Tuple[Text, Text]:
    """Hook specification for getting plugin version info."""


@hookspec  # type: ignore[misc]
def configure_commandline(cmdline_arguments: argparse.Namespace) -> Optional[Text]:
    """Hook specification for configuring plugin CLI."""


@hookspec  # type: ignore[misc]
def init_telemetry(endpoints_file: Optional[Text]) -> None:
    """Hook specification for initialising plugin telemetry."""


@hookspec  # type: ignore[misc]
def mock_tracker_for_evaluation(
    example: Message, model_metadata: ModelMetadata
) -> DialogueStateTracker:
    """Generate a mocked tracker for NLU evaluation."""


@hookspec
def clean_entity_targets_for_evaluation(
    merged_targets: List[str], extractor: str
) -> List[str]:
    """Remove entity targets for space-based entity extractors."""
