---
sidebar_label: rasa.nlu.extractors.duckling_entity_extractor
title: rasa.nlu.extractors.duckling_entity_extractor
---
## DucklingEntityExtractor Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class DucklingEntityExtractor(GraphComponent,  EntityExtractorMixin)
```

Searches for structured entities, e.g. dates, using a duckling server.

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any]) -> None
```

Creates the extractor.

**Arguments**:

- `config` - The extractor&#x27;s config.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> DucklingEntityExtractor
```

Creates component (see parent class for full docstring).

#### process

```python
 | process(messages: List[Message]) -> List[Message]
```

Augments the message with potentially extracted entities.

