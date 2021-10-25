---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.mitie_featurizer
title: rasa.nlu.featurizers.dense_featurizer.mitie_featurizer
---
## MitieFeaturizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER,
    is_trainable=False,
    model_from="MitieNLP",
)
class MitieFeaturizer(DenseFeaturizer,  GraphComponent)
```

A class that featurizes using Mitie.

#### required\_components

```python
@classmethod
def required_components(cls) -> List[Type]
```

Components that should be included in the pipeline before this component.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns the component&#x27;s default config.

#### required\_packages

```python
@staticmethod
def required_packages() -> List[Text]
```

Any extra python dependencies required for this component to run.

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], execution_context: ExecutionContext) -> None
```

Instantiates a new `MitieFeaturizer` instance.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> MitieFeaturizer
```

Creates a new untrained component (see parent class for full docstring).

#### validate\_config

```python
@classmethod
def validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

#### ndim

```python
def ndim(feature_extractor: "mitie.total_word_feature_extractor") -> int
```

Returns the number of dimensions.

#### process

```python
def process(messages: List[Message], model: MitieModel) -> List[Message]
```

Featurizes all given messages in-place.

**Returns**:

  The given list of messages which have been modified in-place.

#### process\_training\_data

```python
def process_training_data(training_data: TrainingData, model: MitieModel) -> TrainingData
```

Processes the training examples in the given training data in-place.

**Arguments**:

- `training_data` - Training data.
- `model` - A Mitie model.
  

**Returns**:

  Same training data after processing.

#### features\_for\_tokens

```python
def features_for_tokens(tokens: List[Token], feature_extractor: "mitie.total_word_feature_extractor") -> Tuple[np.ndarray, np.ndarray]
```

Calculates features.

