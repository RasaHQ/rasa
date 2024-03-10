import pytest
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.recipes.recipe import Recipe, InvalidRecipeException


def test_invalid_recipe():
    with pytest.raises(InvalidRecipeException):
        Recipe.recipe_for_name("dalksldkas")


def test_recipe_is_none():
    with pytest.warns(FutureWarning):
        recipe = Recipe.recipe_for_name(None)

    assert isinstance(recipe, DefaultV1Recipe)
