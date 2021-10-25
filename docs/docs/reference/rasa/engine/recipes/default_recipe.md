---
sidebar_label: rasa.engine.recipes.default_recipe
title: rasa.engine.recipes.default_recipe
---
## DefaultV1RecipeRegisterException Objects

```python
class DefaultV1RecipeRegisterException(RasaException)
```

If you register a class which is not of type `GraphComponent`.

## DefaultV1Recipe Objects

```python
class DefaultV1Recipe(Recipe)
```

Recipe which converts the normal model config to train and predict graph.

## ComponentType Objects

```python
@enum.unique
class ComponentType(Enum)
```

Enum to categorize and place custom components correctly in the graph.

#### \_\_init\_\_

```python
def __init__() -> None
```

Creates recipe.

## RegisteredComponent Objects

```python
@dataclasses.dataclass()
class RegisteredComponent()
```

Describes a graph component which was registered with the decorator.

#### register

```python
@classmethod
def register(cls, component_types: Union[ComponentType, List[ComponentType]], is_trainable: bool, model_from: Optional[Text] = None) -> Callable[[Type[GraphComponent]], Type[GraphComponent]]
```

This decorator can be used to register classes with the recipe.

**Arguments**:

- `component_types` - Describes the types of a component which are then used
  to place the component in the graph.
- `is_trainable` - `True` if the component requires training.
- `model_from` - Can be used if this component requires a pre-loaded model
  such as `SpacyNLP` or `MitieNLP`.
  

**Returns**:

  The registered class.

#### graph\_config\_for\_recipe

```python
def graph_config_for_recipe(config: Dict, cli_parameters: Dict[Text, Any], training_type: TrainingType = TrainingType.BOTH, is_finetuning: bool = False) -> GraphModelConfiguration
```

Converts the default config to graphs (see interface for full docstring).

