import logging

from rasa.engine.recipes.recipe import Recipe
from rasa.engine.graph import GraphModelConfiguration
from rasa.shared.constants import DOCS_URL_GRAPH_RECIPE
from rasa.shared.data import TrainingType
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.utils.common import mark_as_experimental_feature
from rasa.shared.utils.io import raise_warning
from rasa.engine.graph import GraphSchema

from typing import Dict, Text, Any, Tuple


logger = logging.getLogger(__name__)


class GraphV1Recipe(Recipe):
    """Recipe which converts the graph model config to train and predict graph."""

    name = "graph.v1"

    def get_targets(
        self, config: Dict, training_type: TrainingType
    ) -> Tuple[Text, Any]:
        """Return NLU and core targets from config dictionary.

        Note that default recipe has `nlu_target` and `core_target` as
        fixed values of `run_RegexMessageHandler` and `select_prediction`
        respectively. For graph recipe, target values are customizable. These
        can be used in validation (default recipe does this validation check)
        and during execution (all recipes use targets during execution).
        """
        if training_type == TrainingType.NLU:
            core_required = False
            core_target = None
        else:
            core_required = True
            core_target = config.get("core_target")
        # NLU target is required because core (prediction) depends on NLU.
        nlu_target = config.get("nlu_target")
        if nlu_target is None or (core_required and core_target is None):
            raise InvalidConfigException(
                "Can't find target names for NLU and/or core. Please make "
                "sure to provide 'nlu_target' (required for all training types) "
                "and 'core_target' (required if training is not just NLU) values in "
                "your config.yml file."
            )
        return nlu_target, core_target

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

        nlu_target, core_target = self.get_targets(config, training_type)

        return GraphModelConfiguration(
            train_schema=GraphSchema.from_dict(config.get("train_schema")),
            predict_schema=GraphSchema.from_dict(config.get("predict_schema")),
            training_type=training_type,
            language=config.get("language"),
            core_target=core_target,
            nlu_target=nlu_target,
        )
