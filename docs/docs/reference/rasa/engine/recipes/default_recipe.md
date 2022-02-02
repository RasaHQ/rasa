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
 | __init__() -> None
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
 | @classmethod
 | register(cls, component_types: Union[ComponentType, List[ComponentType]], is_trainable: bool, model_from: Optional[Text] = None) -> Callable[[Type[GraphComponent]], Type[GraphComponent]]
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
 | graph_config_for_recipe(config: Dict, cli_parameters: Dict[Text, Any], training_type: TrainingType = TrainingType.BOTH, is_finetuning: bool = False) -> GraphModelConfiguration
```

Converts the default config to graphs (see interface for full docstring).

#### auto\_configure

```python
 | @staticmethod
 | auto_configure(config_file_path: Optional[Text], config: Dict, training_type: Optional[TrainingType] = TrainingType.BOTH) -> Tuple[Dict[Text, Any], Set[str], Set[str]]
```

Determine configuration from auto-filled configuration file.

Keys that are provided and have a value in the file are kept. Keys that are not
provided are configured automatically.

Note that this needs to be called explicitly; ie. we cannot
auto-configure automatically from importers because importers are not
allowed to access code outside of `rasa.shared`.

**Arguments**:

- `config_file_path` - The path to the configuration file.
- `config` - Configuration in dictionary format.
- `training_type` - Optional training type to auto-configure. By default
  both core and NLU will be auto-configured.

#### complete\_config

```python
 | @staticmethod
 | complete_config(config: Dict[Text, Any], keys_to_configure: Set[Text]) -> Dict[Text, Any]
```

Complete a config by adding automatic configuration for the specified keys.

**Arguments**:

- `config` - The provided configuration.
- `keys_to_configure` - Keys to be configured automatically (e.g. `policies`).
  

**Returns**:

  The resulting configuration including both the provided and
  the automatically configured keys.

