---
sidebar_label: rasa.nlu.classifiers.fallback_classifier
title: rasa.nlu.classifiers.fallback_classifier
---

## FallbackClassifier Objects

```python
class FallbackClassifier(IntentClassifier)
```

#### process

```python
 | process(message: Message, **kwargs: Any) -> None
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

