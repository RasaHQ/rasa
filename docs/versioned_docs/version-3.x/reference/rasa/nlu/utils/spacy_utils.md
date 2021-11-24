---
sidebar_label: rasa.nlu.utils.spacy_utils
title: rasa.nlu.utils.spacy_utils
---
## SpacyModel Objects

```python
@dataclasses.dataclass
class SpacyModel()
```

Wraps `SpacyNLP` output to make it fingerprintable.

#### fingerprint

```python
 | fingerprint() -> Text
```

Fingerprints the model name.

Use a static fingerprint as we assume this only changes if the model name
changes and want to avoid investigating the model in greater detail for now.

**Returns**:

  Fingerprint for model.

## SpacyNLP Objects

```python
@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.MODEL_LOADER,
        DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER,
    ],
    is_trainable=False,
    model_from="SpacyNLP",
)
class SpacyNLP(GraphComponent)
```

Component which provides the common loaded SpaCy model to others.

This is used to avoid loading the SpaCy model multiple times. Instead the Spacy
model is only loaded once and then shared by depending components.

#### \_\_init\_\_

```python
 | __init__(model: SpacyModel, config: Dict[Text, Any]) -> None
```

Initializes a `SpacyNLP`.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

Default config.

#### load\_model

```python
 | @staticmethod
 | load_model(spacy_model_name: Text) -> SpacyModel
```

Try loading the model, catching the OSError if missing.

#### required\_packages

```python
 | @classmethod
 | required_packages(cls) -> List[Text]
```

Lists required dependencies (see parent class for full docstring).

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> SpacyNLP
```

Creates component (see parent class for full docstring).

#### ensure\_proper\_language\_model

```python
 | @staticmethod
 | ensure_proper_language_model(nlp: Optional[Language]) -> None
```

Checks if the SpaCy language model is properly loaded.

Raises an exception if the model is invalid.

#### provide

```python
 | provide() -> SpacyModel
```

Provides the loaded SpaCy model.

#### process\_training\_data

```python
 | process_training_data(training_data: TrainingData, model: SpacyModel) -> TrainingData
```

Adds SpaCy tokens and features to training data messages.

#### process

```python
 | process(messages: List[Message], model: SpacyModel) -> List[Message]
```

Adds SpaCy tokens and features to messages.

