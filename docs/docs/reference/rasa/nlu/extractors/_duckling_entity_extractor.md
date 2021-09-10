---
sidebar_label: rasa.nlu.extractors._duckling_entity_extractor
title: rasa.nlu.extractors._duckling_entity_extractor
---
## DucklingEntityExtractor Objects

```python
class DucklingEntityExtractor(EntityExtractor)
```

Searches for structured entites, e.g. dates, using a duckling server.

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional[Metadata] = None, cached_component: Optional["DucklingEntityExtractor"] = None, **kwargs: Any, ,) -> "DucklingEntityExtractor"
```

Loads trained component (see parent class for full docstring).

