import logging

from rasa.engine.recipes.recipe import Recipe
from rasa.engine.graph import GraphModelConfiguration
from rasa.shared.constants import DOCS_URL_GRAPH_RECIPE
from rasa.shared.data import TrainingType
from rasa.shared.utils.common import mark_as_experimental_feature
from rasa.shared.utils.io import raise_warning
from rasa.engine.graph import GraphSchema
from rasa.nlu.classifiers.regex_message_handler import RegexMessageHandler

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
        if cli_parameters or is_finetuning:
            raise_warning(
                "Unlike the Default Recipe, Graph Recipe does not utilize CLI "
                "parameters or finetuning and these configurations will be ignored. "
                "Add configuration to the recipe itself if you want them to be used.",
                docs=DOCS_URL_GRAPH_RECIPE,
            )
        # Note that default recipe has `nlu_target` and `core_target` as
        # fixed values of `run_RegexMessageHandler` and `select_prediction`
        # respectively. For graph recipe, target values are customizable. These
        # can be used in validation (default recipe does this validation check)
        # and during execution (all recipes use targets during execution).
        if training_type == TrainingType.NLU:
            core_target = None
        else:
            core_target = config.get("core_target", "select_prediction")
        # there's always an NLU target because core (prediction) always needs NLU.
        nlu_target = config.get("nlu_target", f"run_{RegexMessageHandler.__name__}")

        return GraphModelConfiguration(
            train_schema=GraphSchema.from_dict(config.get("train_schema")),
            predict_schema=GraphSchema.from_dict(config.get("predict_schema")),
            training_type=training_type,
            language=config.get("language"),
            core_target=core_target,
            nlu_target=nlu_target,
        )
