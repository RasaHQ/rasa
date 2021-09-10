---
sidebar_label: rasa.nlu.classifiers._fallback_classifier
title: rasa.nlu.classifiers._fallback_classifier
---
## FallbackClassifier Objects

```python
class FallbackClassifier(IntentClassifier)
```

#### process

```python
def process(message: Message, **kwargs: Any) -> None
```

Process an incoming message.

This is the components chance to process an incoming
message. The component can rely on
any context attribute to be present, that gets created
by a call to :meth:`rasa.nlu.components.Component.create`
of ANY component and
on any context attributes created by a call to
:meth:`rasa.nlu.components.Component.process`
of components previous to this one.

**Arguments**:

- `message` - The :class:`rasa.shared.nlu.training_data.message.Message` to
  process.

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

