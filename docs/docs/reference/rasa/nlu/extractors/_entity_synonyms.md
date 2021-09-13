---
sidebar_label: rasa.nlu.extractors._entity_synonyms
title: rasa.nlu.extractors._entity_synonyms
---
## EntitySynonymMapper Objects

```python
class EntitySynonymMapper(EntityExtractor)
```

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional[Metadata] = None, cached_component: Optional["EntitySynonymMapper"] = None, **kwargs: Any, ,) -> "EntitySynonymMapper"
```

Loads trained component (see parent class for full docstring).

