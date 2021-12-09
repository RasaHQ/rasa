---
sidebar_label: rasa.nlu.featurizers.sparse_featurizer.regex_featurizer
title: rasa.nlu.featurizers.sparse_featurizer.regex_featurizer
---
## RegexFeaturizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True
)
class RegexFeaturizer(SparseFeaturizer,  GraphComponent)
```

Adds message features based on regex expressions.

#### required\_components

```python
 | @classmethod
 | required_components(cls) -> List[Type]
```

Components that should be included in the pipeline before this component.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

Returns the component&#x27;s default config.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, known_patterns: Optional[List[Dict[Text, Text]]] = None) -> None
```

Constructs new features for regexes and lookup table using regex expressions.

**Arguments**:

- `config` - Configuration for the component.
- `model_storage` - Storage which graph components can use to persist and load
  themselves.
- `resource` - Resource locator for this component which can be used to persist
  and load itself from the `model_storage`.
- `execution_context` - Information about the current graph run.
- `known_patterns` - Regex Patterns the component should pre-load itself with.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> RegexFeaturizer
```

Creates a new untrained component (see parent class for full docstring).

#### train

```python
 | train(training_data: TrainingData) -> Resource
```

Trains the component with all patterns extracted from training data.

#### process\_training\_data

```python
 | process_training_data(training_data: TrainingData) -> TrainingData
```

Processes the training examples (see parent class for full docstring).

#### process

```python
 | process(messages: List[Message]) -> List[Message]
```

Featurizes all given messages in-place.

**Returns**:

  the given list of messages which have been modified in-place

#### load

```python
 | @classmethod
 | load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> RegexFeaturizer
```

Loads trained component (see parent class for full docstring).

#### validate\_config

```python
 | @classmethod
 | validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

