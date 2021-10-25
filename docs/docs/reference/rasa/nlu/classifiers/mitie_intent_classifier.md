---
sidebar_label: rasa.nlu.classifiers.mitie_intent_classifier
title: rasa.nlu.classifiers.mitie_intent_classifier
---
## MitieIntentClassifier Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
    is_trainable=True,
    model_from="MitieNLP",
)
class MitieIntentClassifier(GraphComponent,  IntentClassifier)
```

Intent classifier which uses the `mitie` library.

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

Returns default config (see parent class for full docstring).

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, clf: Optional["mitie.text_categorizer"] = None) -> None
```

Constructs a new intent classifier using the MITIE framework.

#### required\_packages

```python
 | @classmethod
 | required_packages(cls) -> List[Text]
```

Lists required dependencies (see parent class for full docstring).

#### train

```python
 | train(training_data: TrainingData, model: MitieModel) -> Resource
```

Trains classifier.

**Arguments**:

- `training_data` - The NLU training data.
- `model` - The loaded mitie model provided by `MitieNLP`.
  

**Returns**:

  The resource locator for the trained classifier.

#### process

```python
 | process(messages: List[Message], model: MitieModel) -> List[Message]
```

Make intent predictions using `mitie`.

**Arguments**:

- `messages` - The message which the intents should be predicted for.
- `model` - The loaded mitie model provided by `MitieNLP`.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> MitieIntentClassifier
```

Creates component for training see parent class for full docstring).

#### load

```python
 | @classmethod
 | load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> MitieIntentClassifier
```

Loads component for inference see parent class for full docstring).

