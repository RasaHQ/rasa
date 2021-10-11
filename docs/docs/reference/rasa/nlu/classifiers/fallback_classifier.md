---
sidebar_label: rasa.nlu.classifiers.fallback_classifier
title: rasa.nlu.classifiers.fallback_classifier
---
## FallbackClassifierGraphComponent Objects

```python
class FallbackClassifierGraphComponent(GraphComponent,  IntentClassifier2)
```

Handles incoming messages with low NLU confidence.

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

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any]) -> None
```

Constructs a new fallback classifier.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> FallbackClassifierGraphComponent
```

Creates a new component (see parent class for full docstring).

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Process a list of incoming messages.

This is the component&#x27;s chance to process incoming
messages. The component can rely on
any context attribute to be present, that gets created
by a call to :meth:`rasa.nlu.components.Component.create`
of ANY component and
on any context attributes created by a call to
:meth:`rasa.nlu.components.Component.process`
of components previous to this one.

**Arguments**:

- `messages` - List containing :class:
  `rasa.shared.nlu.training_data.message.Message` to process.

#### is\_fallback\_classifier\_prediction

```python
def is_fallback_classifier_prediction(prediction: Dict[Text, Any]) -> bool
```

Checks if the intent was predicted by the `FallbackClassifier`.

**Arguments**:

- `prediction` - The prediction of the NLU model.
  

**Returns**:

  `True` if the top classified intent was the fallback intent.

#### undo\_fallback\_prediction

```python
def undo_fallback_prediction(prediction: Dict[Text, Any]) -> Dict[Text, Any]
```

Undo the prediction of the fallback intent.

**Arguments**:

- `prediction` - The prediction of the NLU model.
  

**Returns**:

  The prediction as if the `FallbackClassifier` wasn&#x27;t present in the pipeline.
  If the fallback intent is the only intent, return the prediction as it was
  provided.

