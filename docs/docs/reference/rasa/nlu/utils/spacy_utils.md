---
sidebar_label: rasa.nlu.utils.spacy_utils
title: rasa.nlu.utils.spacy_utils
---
## SpacyModel Objects

```python
@dataclasses.dataclass
class SpacyModel()
```

Wraps `SpacyModelProvider` output to make it fingerprintable.

#### fingerprint

```python
 | fingerprint() -> Text
```

Fingerprints the model name.

Use a static fingerprint as we assume this only changes if the model name
changes and want to avoid investigating the model in greater detail for now.

**Returns**:

  Fingerprint for model.

## SpacyModelProvider Objects

```python
class SpacyModelProvider(GraphComponent)
```

Component which provides the common loaded SpaCy model to others.

This is used to avoid loading the SpaCy model multiple times. Instead the Spacy
model is only loaded once and then shared by depending components.

#### \_\_init\_\_

```python
 | __init__(model: Optional[Language] = None, model_name: Optional[Text] = None) -> None
```

Initializes a `SpacyModelProvider`.

#### load\_model

```python
 | @staticmethod
 | load_model(spacy_model_name: Text) -> Language
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
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> SpacyModelProvider
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

## SpacyPreprocessor Objects

```python
class SpacyPreprocessor(GraphComponent)
```

Processes messages using SpaCy for use by SpacyTokenizer and SpacyFeaturizer.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

Default config for SpacyPreprocessor.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any]) -> None
```

Initializes a `SpacyPreprocessor`.

#### required\_packages

```python
 | @classmethod
 | required_packages(cls) -> List[Text]
```

Lists required dependencies (see parent class for full docstring).

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> SpacyPreprocessor
```

Creates component for training see parent class for full docstring).

#### process\_training\_data

```python
 | process_training_data(training_data: TrainingData, spacy_model: SpacyModel) -> None
```

Adds SpaCy tokens and features to training data messages.

#### process

```python
 | process(messages: List[Message], spacy_model: SpacyModel) -> None
```

Adds SpaCy tokens and features to messages.

