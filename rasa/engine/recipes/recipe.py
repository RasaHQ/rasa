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
    """Base class for `Recipe`s which convert configs to graph schemas."""

    @staticmethod
    def recipe_for_name(name: Optional[Text]) -> Recipe:
        """Returns `Recipe` based on an optional recipe identifier.

        Args:
            name: The identifier which is used to select a certain `Recipe`. If `None`
                the default recipe will be used.

        Returns:
            A recipe which can be used to convert a given config to train and predict
            graph schemas.
        """
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
    ) -> Tuple[GraphSchema, GraphSchema]:
        """Converts a given config to graph schemas for training and prediction.

        Args:
            config: The given config.
            cli_parameters: Potentially passed CLI parameters which need to be
                inserted into certain components config.
            training_type: The given training type which might be used to omit / add
                certain subgraphs.

        Returns:
            A graph schema to train a model and a graph schema to make predictions
            with this model after training.
        """
        raise NotImplementedError()
