---
sidebar_label: rasa.nlu.featurizers.dense_featurizer._convert_featurizer
title: rasa.nlu.featurizers.dense_featurizer._convert_featurizer
---
## ConveRTFeaturizer Objects

```python
class ConveRTFeaturizer(DenseFeaturizer)
```

Featurizer using ConveRT model.

Loads the ConveRT(https://github.com/PolyAI-LDN/polyai-models#convert)
model from TFHub and computes sentence and sequence level feature representations
for dense featurizable attributes of each message object.

#### required\_components

```python
 | @classmethod
 | required_components(cls) -> List[Type[Component]]
```

Components that should be included in the pipeline before this component.

#### required\_packages

```python
 | @classmethod
 | required_packages(cls) -> List[Text]
```

Packages needed to be installed.

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None) -> None
```

Initializes ConveRTFeaturizer with the model and different
encoding signatures.

**Arguments**:

- `component_config` - Configuration for the component.

#### train

```python
 | train(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Featurize all message attributes in the training data with the ConveRT model.

**Arguments**:

- `training_data` - Training data to be featurized
- `config` - Pipeline configuration
- `**kwargs` - Any other arguments.

#### process

```python
 | process(message: Message, **kwargs: Any) -> None
```

Featurize an incoming message with the ConveRT model.

**Arguments**:

- `message` - Message to be featurized
- `**kwargs` - Any other arguments.

#### cache\_key

```python
 | @classmethod
 | cache_key(cls, component_meta: Dict[Text, Any], model_metadata: Metadata) -> Optional[Text]
```

Cache the component for future use.

**Arguments**:

- `component_meta` - configuration for the component.
- `model_metadata` - configuration for the whole pipeline.
  
- `Returns` - key of the cache for future retrievals.

#### provide\_context

```python
 | provide_context() -> Dict[Text, Any]
```

Store the model in pipeline context for future use.

#### tokenize

```python
 | tokenize(message: Message, attribute: Text) -> List[Token]
```

Tokenize the text using the ConveRT model.

ConveRT adds a special char in front of (some) words and splits words into
sub-words. To ensure the entity start and end values matches the token values,
reuse the tokens that are already assigned to the message. If individual tokens
are split up into multiple tokens, add this information to the
respected tokens.

