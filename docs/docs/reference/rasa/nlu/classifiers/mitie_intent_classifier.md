---
sidebar_label: rasa.nlu.classifiers.mitie_intent_classifier
title: rasa.nlu.classifiers.mitie_intent_classifier
---
## MitieIntentClassifierGraphComponent Objects

```python
class MitieIntentClassifierGraphComponent(GraphComponent)
```

Intent classifier which uses the `mitie` library.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

Returns default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, clf: Optional["mitie.text_categorizer"] = None) -> None
```

Constructs a new intent classifier using the MITIE framework.

#### required\_packages

```python
@classmethod
def required_packages(cls) -> List[Text]
```

Lists required dependencies (see parent class for full docstring).

#### train

```python
def train(training_data: TrainingData, model: MitieModel) -> Resource
```

Trains classifier.

**Arguments**:

- `training_data` - The NLU training data.
- `model` - The loaded mitie model provided by `MitieNLP`.
  

**Returns**:

  The resource locator for the trained classifier.

#### process

```python
def process(messages: List[Message], model: MitieModel) -> None
```

Make intent predictions using `mitie`.

**Arguments**:

- `messages` - The message which the intents should be predicted for.
- `model` - The loaded mitie model provided by `MitieNLP`.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> MitieIntentClassifierGraphComponent
```

Creates component for training see parent class for full docstring).

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> MitieIntentClassifierGraphComponent
```

Loads component for inference see parent class for full docstring).

