import logging

from rasa.engine.recipes.recipe import Recipe
from rasa.engine.graph import GraphModelConfiguration
from rasa.shared.data import TrainingType
from rasa.shared.utils.common import mark_as_experimental_feature
from rasa.shared.utils.io import raise_warning
from rasa.engine.graph import GraphSchema
from rasa.nlu.classifiers.regex_message_handler import RegexMessageHandler

from typing import Dict, Text, Any, Optional


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
        if cli_parameters or is_finetuning:
            raise_warning(
                "Unlike the Default Recipe, Graph Recipe does not utilize CLI "
                "parameters or finetuning and these configurations will be ignored. "
                "Add configuration to the recipe itself if you want them to be used.",
                docs="PLACEHOLDER_FOR_DOCS",
            )
        # TODO: Add core_target and nlu_target to graph configuration options so we
        # don't force graph node names to be `run_RegexMessageHandler` and
        # `select_prediction`. These are used in validation and during execution.
        core_target = None if training_type == TrainingType.NLU else "select_prediction"

        return GraphModelConfiguration(
            train_schema=GraphSchema.from_dict(config.get("train_schema")),
            predict_schema=GraphSchema.from_dict(config.get("predict_schema")),
            training_type=training_type,
            language=config.get("language"),
            core_target=core_target,
            # there's always an NLU target because core (prediction) always needs NLU.
            nlu_target=f"run_{RegexMessageHandler.__name__}",
        )
