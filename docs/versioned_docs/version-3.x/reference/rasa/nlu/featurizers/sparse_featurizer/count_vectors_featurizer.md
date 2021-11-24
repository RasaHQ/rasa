---
sidebar_label: rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer
title: rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer
---
## CountVectorsFeaturizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True
)
class CountVectorsFeaturizer(SparseFeaturizer,  GraphComponent)
```

Creates a sequence of token counts features based on sklearn&#x27;s `CountVectorizer`.

All tokens which consist only of digits (e.g. 123 and 99
but not ab12d) will be represented by a single feature.

Set `analyzer` to &#x27;char_wb&#x27;
to use the idea of Subword Semantic Hashing
from https://arxiv.org/abs/1810.07150.

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

#### required\_packages

```python
 | @staticmethod
 | required_packages() -> List[Text]
```

Any extra python dependencies required for this component to run.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, vectorizers: Optional[Dict[Text, "CountVectorizer"]] = None, oov_token: Optional[Text] = None, oov_words: Optional[List[Text]] = None) -> None
```

Constructs a new count vectorizer using the sklearn framework.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> CountVectorsFeaturizer
```

Creates a new untrained component (see parent class for full docstring).

#### train

```python
 | train(training_data: TrainingData, model: Optional[SpacyModel] = None) -> Resource
```

Trains the featurizer.

Take parameters from config and
construct a new count vectorizer using the sklearn framework.

#### process\_training\_data

```python
 | process_training_data(training_data: TrainingData) -> TrainingData
```

Processes the training examples in the given training data in-place.

**Arguments**:

- `training_data` - the training data
  

**Returns**:

  same training data after processing

#### process

```python
 | process(messages: List[Message]) -> List[Message]
```

Processes incoming message and compute and set features.

#### persist

```python
 | persist() -> None
```

Persist this model into the passed directory.

Returns the metadata necessary to load the model again.

#### load

```python
 | @classmethod
 | load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> CountVectorsFeaturizer
```

Loads trained component (see parent class for full docstring).

#### validate\_config

```python
 | @classmethod
 | validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

