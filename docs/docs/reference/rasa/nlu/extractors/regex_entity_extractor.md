---
sidebar_label: rasa.nlu.extractors.regex_entity_extractor
title: rasa.nlu.extractors.regex_entity_extractor
---
## RegexEntityExtractorGraphComponent Objects

```python
class RegexEntityExtractorGraphComponent(GraphComponent,  EntityExtractorMixin)
```

Extracts entities via lookup tables and regexes defined in the training data.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> GraphComponent
```

Creates a new `GraphComponent`.

**Arguments**:

- `config` - This config overrides the `default_config`.
- `model_storage` - Storage which graph components can use to persist and load
  themselves.
- `resource` - Resource locator for this component which can be used to persist
  and load itself from the `model_storage`.
- `execution_context` - Information about the current graph run. Unused.
  
- `Returns` - An instantiated `GraphComponent`.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, patterns: Optional[List[Dict[Text, Text]]] = None) -> None
```

Creates a new instance.

**Arguments**:

- `config` - The configuration.
- `model_storage` - Storage which graph components can use to persist and load
  themselves.
- `resource` - Resource locator for this component which can be used to persist
  and load itself from the `model_storage`.
- `patterns` - a list of patterns

#### train

```python
 | train(training_data: TrainingData) -> Resource
```

Extract patterns from the training data.

**Arguments**:

- `training_data` - the training data

#### process

```python
 | process(messages: List[Message]) -> List[Message]
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
 | @classmethod
 | load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> RegexEntityExtractorGraphComponent
```

Loads trained component (see parent class for full docstring).

#### persist

```python
 | persist() -> None
```

Persist this model.

