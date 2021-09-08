---
sidebar_label: rasa.nlu.extractors.spacy_entity_extractor
title: rasa.nlu.extractors.spacy_entity_extractor
---
## SpacyEntityExtractorGraphComponent Objects

```python
class SpacyEntityExtractorGraphComponent(GraphComponent,  EntityExtractorMixin)
```

Entity extractor which uses SpaCy.

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

Initialize SpacyEntityExtractorGraphComponent.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> GraphComponent
```

Creates a new component (see parent class for full docstring).

#### required\_packages

```python
 | @classmethod
 | required_packages(cls) -> List[Text]
```

Lists required dependencies (see parent class for full docstring).

#### process

```python
 | process(messages: List[Message], model: SpacyModel) -> List[Message]
```

Extract entities using SpaCy.

**Arguments**:

- `messages` - List of messages to process.
- `model` - Container holding a loaded spacy nlp model.
  
- `Returns` - The processed messages.

