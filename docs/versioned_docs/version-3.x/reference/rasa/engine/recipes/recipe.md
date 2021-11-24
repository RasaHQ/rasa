---
sidebar_label: rasa.engine.recipes.recipe
title: rasa.engine.recipes.recipe
---
## InvalidRecipeException Objects

```python
class InvalidRecipeException(RasaException)
```

Exception in case the specified recipe is invalid.

## Recipe Objects

```python
class Recipe(abc.ABC)
```

Base class for `Recipe`s which convert configs to graph schemas.

#### recipe\_for\_name

```python
 | @staticmethod
 | recipe_for_name(name: Optional[Text]) -> Recipe
```

Returns `Recipe` based on an optional recipe identifier.

**Arguments**:

- `name` - The identifier which is used to select a certain `Recipe`. If `None`
  the default recipe will be used.
  

**Returns**:

  A recipe which can be used to convert a given config to train and predict
  graph schemas.

#### graph\_config\_for\_recipe

```python
 | @abc.abstractmethod
 | graph_config_for_recipe(config: Dict, cli_parameters: Dict[Text, Any], training_type: TrainingType = TrainingType.BOTH, is_finetuning: bool = False) -> GraphModelConfiguration
```

Converts a config to a graph compatible model configuration.

**Arguments**:

- `config` - The config which the `Recipe` is supposed to convert.
- `cli_parameters` - Potential CLI params which should be interpolated into the
  components configs.
- `training_type` - The current training type. Can be used to omit / add certain
  parts of the graphs.
- `is_finetuning` - If `True` then the components should load themselves from
  trained version of themselves instead of using `create` to start from
  scratch.
  

**Returns**:

  The model configuration which enables to run the model as a graph for
  training and prediction.

