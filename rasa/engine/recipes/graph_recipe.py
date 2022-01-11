from rasa.engine.recipes.recipe import Recipe
from rasa.engine.graph import (
    GraphSchema,
    GraphComponent,
    SchemaNode,
    GraphModelConfiguration,
)
from rasa.shared.data import TrainingType

from typing import Dict, Text, Any, Tuple, Type, Optional, List, Callable, Set, Union


class GraphV1Recipe(Recipe):
    """Recipe which converts the graph model config to train and predict graph."""

    name = "graph.v1"

    def graph_config_for_recipe(
        self,
        config: Dict,
        cli_parameters: Dict[Text, Any],
        training_type: TrainingType = TrainingType.BOTH,
        is_finetuning: bool = False,
    ) -> GraphModelConfiguration:
        """Converts the default config to graphs (see interface for full docstring)."""
