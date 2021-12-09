---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.spacy_featurizer
title: rasa.nlu.featurizers.dense_featurizer.spacy_featurizer
---
## SpacyFeaturizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class SpacyFeaturizer(DenseFeaturizer,  GraphComponent)
```

Featurize messages using SpaCy.

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

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], name: Text) -> None
```

Initializes SpacyFeaturizer.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> GraphComponent
```

Creates a new component (see parent class for full docstring).

#### process

```python
 | process(messages: List[Message]) -> List[Message]
```

Processes incoming messages and computes and sets features.

#### process\_training\_data

```python
 | process_training_data(training_data: TrainingData) -> TrainingData
```

Processes the training examples in the given training data in-place.

**Arguments**:

- `training_data` - Training data.
  

**Returns**:

  Same training data after processing.

#### validate\_config

```python
 | @classmethod
 | validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

