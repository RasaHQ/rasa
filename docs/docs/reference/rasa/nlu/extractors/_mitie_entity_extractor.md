---
sidebar_label: rasa.nlu.extractors._mitie_entity_extractor
title: rasa.nlu.extractors._mitie_entity_extractor
---
## MitieEntityExtractor Objects

```python
class MitieEntityExtractor(EntityExtractor)
```

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None, ner: Optional["mitie.named_entity_extractor"] = None) -> None
```

Construct a new intent classifier using the sklearn framework.

#### load

```python
 | @classmethod
 | load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Metadata = None, cached_component: Optional["MitieEntityExtractor"] = None, **kwargs: Any, ,) -> "MitieEntityExtractor"
```

Loads trained component (see parent class for full docstring).

