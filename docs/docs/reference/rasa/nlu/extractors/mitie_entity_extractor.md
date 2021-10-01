---
sidebar_label: rasa.nlu.extractors.mitie_entity_extractor
title: rasa.nlu.extractors.mitie_entity_extractor
---
## MitieEntityExtractorGraphComponent Objects

```python
class MitieEntityExtractorGraphComponent(GraphComponent,  EntityExtractorMixin)
```

A Mitie Entity Extractor (which is a thin wrapper around `Dlib-ml`).

#### required\_packages

```python
@staticmethod
def required_packages() -> List[Text]
```

Any extra python dependencies required for this component to run.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, ner: Optional["mitie.named_entity_extractor"] = None) -> None
```

Creates a new instance.

**Arguments**:

- `config` - The configuration.
- `model_storage` - Storage which graph components can use to persist and load
  themselves.
- `resource` - Resource locator for this component which can be used to persist
  and load itself from the `model_storage`.
- `ner` - Mitie named entity extractor

#### validate\_config

```python
def validate_config(cls, config: Dict[Text, Any]) -> None
```

Checks whether the given configuration is valid.

**Arguments**:

- `config` - a configuration for a Mitie entity extractor component

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> GraphComponent
```

Creates a new `MitieEntityExtractorGraphComponent`.

**Arguments**:

- `config` - This config overrides the `default_config`.
- `model_storage` - Storage which graph components can use to persist and load
  themselves.
- `resource` - Resource locator for this component which can be used to persist
  and load itself from the `model_storage`.
- `execution_context` - Information about the current graph run. Unused.
  
- `Returns` - An instantiated `MitieEntityExtractorGraphComponent`.

#### train

```python
def train(training_data: TrainingData, mitie_model: MitieModel) -> Resource
```

Trains a MITIE named entity recognizer.

**Arguments**:

- `training_data` - the training data
- `mitie_model` - a MitieModel

**Returns**:

  resource for loading the trained model

#### process

```python
def process(messages: List[Message], mitie_model: MitieModel) -> List[Message]
```

Extracts entities from messages and appends them to the attribute.

If no patterns where found during training, then the given messages will not
be modified. In particular, if no `ENTITIES` attribute exists yet, then
it will *not* be created.

If no pattern can be found in the given message, then no entities will be
added to any existing list of entities. However, if no `ENTITIES` attribute
exists yet, then an `ENTITIES` attribute will be created.

**Returns**:

  the given list of messages that have been modified

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> MitieEntityExtractorGraphComponent
```

Loads trained component (see parent class for full docstring).

#### persist

```python
def persist() -> None
```

Persist this model.

