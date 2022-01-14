import logging

from rasa.engine.recipes.recipe import Recipe
from rasa.engine.graph import GraphModelConfiguration
from rasa.shared.data import TrainingType
from rasa.shared.utils.common import mark_as_experimental_feature

from typing import Dict, Text, Any


logger = logging.getLogger(__name__)


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
        mark_as_experimental_feature("graph recipe")
        if cli_parameters:
            logger.warning(
                "Unlike the Default Recipe, Graph Recipe does not utilize CLI "
                "parameters and it will be ignored. Add configuration to the "
                "recipe itself if you want them to be used."
            )
