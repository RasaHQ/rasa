from __future__ import annotations

import abc
from typing import Text, Tuple, Dict, Any, Optional

import rasa.shared.utils.io
from rasa.engine.graph import GraphSchema
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.autoconfig import TrainingType


class InvalidRecipeException(RasaException):
    """Exception in case the specified recipe is invalid."""


class Recipe(abc.ABC):
    @staticmethod
    def recipe_for_name(name: Optional[Text]) -> Recipe:
        from rasa.engine.recipes.default_recipe import DefaultV1Recipe

        if name is None:
            rasa.shared.utils.io.raise_deprecation_warning(
                "From Rasa Open Source 4.0.0 onwards it will be required to specify "
                "a recipe in your model configuration. Defaulting to recipe "
                f"'{DefaultV1Recipe.name}'."
            )
            return DefaultV1Recipe()
        recipes = {DefaultV1Recipe.name: DefaultV1Recipe}

        recipe_constructor = recipes.get(name)
        if recipe_constructor:
            return recipe_constructor()

        raise InvalidRecipeException(
            f"No recipe with name '{name}' was found. "
            f"Available recipes are: "
            f"'{DefaultV1Recipe.name}'."
        )

    @abc.abstractmethod
    def schemas_for_config(
        self,
        config: Dict,
        cli_parameters: Dict[Text, Any],
        training_type: TrainingType = TrainingType.BOTH,
        is_finetuning: bool = False,
    ) -> Tuple[GraphSchema, GraphSchema]:
        """Converts a config to graph schemas for training and prediction.

        Args:
            config: The config which the `Recipe` is supposed to convert.
            cli_parameters: Potential CLI params which should be interpolated into the
                components configs.
            training_type: The current training type. Can be used to omit / add certain
                parts of the graphs.
            is_finetuning: If `True` then the components should load themselves from
                trained version of themselves instead of using `create` to start from
                scratch.

        Returns:
            A schema for training a model and a schema for making predictions with this
            trained model.
        """
        ...
