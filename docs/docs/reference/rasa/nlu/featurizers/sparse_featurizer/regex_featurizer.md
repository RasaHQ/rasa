---
sidebar_label: rasa.nlu.featurizers.sparse_featurizer.regex_featurizer
title: rasa.nlu.featurizers.sparse_featurizer.regex_featurizer
---
## RegexFeaturizerGraphComponent Objects

```python
class RegexFeaturizerGraphComponent(SparseFeaturizer2,  GraphComponent)
```

Adds message features based on regex expressions.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns the component&#x27;s default config.

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, known_patterns: Optional[List[Dict[Text, Text]]] = None) -> None
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
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> RegexFeaturizerGraphComponent
```

Creates a new untrained component (see parent class for full docstring).

#### train

```python
def train(training_data: TrainingData) -> Resource
```

Trains the component with all patterns extracted from training data.

#### process\_training\_data

```python
def process_training_data(training_data: TrainingData) -> TrainingData
```

Processes the training examples (see parent class for full docstring).

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Featurizes all given messages in-place.

**Returns**:

  the given list of messages which have been modified in-place

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> RegexFeaturizerGraphComponent
```

Loads trained component (see parent class for full docstring).

#### validate\_config

```python
@classmethod
def validate_config(cls, config: Dict[Text, Any]) -> None
```

Validates that the component is configured properly.

#### validate\_compatibility\_with\_tokenizer

```python
@classmethod
def validate_compatibility_with_tokenizer(cls, config: Dict[Text, Any], tokenizer_type: Type[Tokenizer]) -> None
```

Validates that the featurizer is compatible with the given tokenizer.

