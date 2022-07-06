---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.convert_featurizer
title: rasa.nlu.featurizers.dense_featurizer.convert_featurizer
---
## ConveRTFeaturizer Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class ConveRTFeaturizer(DenseFeaturizer,  GraphComponent)
```

Featurizer using ConveRT model.

Loads the ConveRT(https://github.com/PolyAI-LDN/polyai-models#convert)
model from TFHub and computes sentence and sequence level feature representations
for dense featurizable attributes of each message object.

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

The component&#x27;s default config (see parent class for full docstring).

#### required\_packages

```python
@staticmethod
def required_packages() -> List[Text]
```

Packages needed to be installed.

#### supported\_languages

```python
@staticmethod
def supported_languages() -> Optional[List[Text]]
```

Determines which languages this component can work with.

Returns: A list of supported languages, or `None` to signify all are supported.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> ConveRTFeaturizer
```

Creates a new component (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(name: Text, config: Dict[Text, Any]) -> None
```

Initializes a `ConveRTFeaturizer`.

**Arguments**:

- `name` - An identifier for this featurizer.
- `config` - The configuration.

#### validate\_config

```python
@classmethod
def validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

#### process\_training\_data

```python
def process_training_data(training_data: TrainingData) -> TrainingData
```

Featurize all message attributes in the training data with the ConveRT model.

**Arguments**:

- `training_data` - Training data to be featurized
  

**Returns**:

  featurized training data

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Featurize an incoming message with the ConveRT model.

**Arguments**:

- `messages` - Message to be featurized

#### tokenize

```python
def tokenize(message: Message, attribute: Text) -> List[Token]
```

Tokenize the text using the ConveRT model.

ConveRT adds a special char in front of (some) words and splits words into
sub-words. To ensure the entity start and end values matches the token values,
reuse the tokens that are already assigned to the message. If individual tokens
are split up into multiple tokens, add this information to the
respected tokens.

