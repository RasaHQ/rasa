---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.mitie_featurizer
title: rasa.nlu.featurizers.dense_featurizer.mitie_featurizer
---
## MitieFeaturizerGraphComponent Objects

```python
class MitieFeaturizerGraphComponent(DenseFeaturizer2,  GraphComponent)
```

A class that featurizes using Mitie.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

Returns the component&#x27;s default config.

#### required\_packages

```python
 | @staticmethod
 | required_packages() -> List[Text]
```

Any extra python dependencies required for this component to run.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], execution_context: ExecutionContext) -> None
```

Instantiates a new `MitieFeaturizerGraphComponent` instance.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> "MitieFeaturizerGraphComponent"
```

Creates a new untrained component (see parent class for full docstring).

#### validate\_config

```python
 | @classmethod
 | validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

#### validate\_compatibility\_with\_tokenizer

```python
 | @classmethod
 | validate_compatibility_with_tokenizer(cls, config: Dict[Text, Any], tokenizer_type: Type[Tokenizer]) -> None
```

Validate a configuration for this component in the context of a recipe.

#### ndim

```python
 | ndim(feature_extractor: "mitie.total_word_feature_extractor") -> int
```

Returns the number of dimensions.

#### process

```python
 | process(messages: List[Message], model: MitieModel) -> List[Message]
```

Featurizes all given messages in-place.

**Returns**:

  The given list of messages which have been modified in-place.

#### process\_training\_data

```python
 | process_training_data(training_data: TrainingData, model: MitieModel) -> TrainingData
```

Processes the training examples in the given training data in-place.

**Arguments**:

- `training_data` - Training data.
- `model` - A Mitie model.
  

**Returns**:

  Same training data after processing.

#### features\_for\_tokens

```python
 | features_for_tokens(tokens: List[Token], feature_extractor: "mitie.total_word_feature_extractor") -> Tuple[np.ndarray, np.ndarray]
```

Calculates features.

