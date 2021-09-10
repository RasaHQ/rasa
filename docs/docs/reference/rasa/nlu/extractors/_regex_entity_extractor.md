---
sidebar_label: rasa.nlu.extractors._regex_entity_extractor
title: rasa.nlu.extractors._regex_entity_extractor
---
## RegexEntityExtractor Objects

```python
class RegexEntityExtractor(EntityExtractor)
```

Searches for entities in the user&#x27;s message using the lookup tables and regexes
defined in the training data.

#### \_\_init\_\_

```python
def __init__(component_config: Optional[Dict[Text, Any]] = None, patterns: Optional[List[Dict[Text, Text]]] = None)
```

Extracts entities using the lookup tables and/or regexes defined.

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional[Metadata] = None, cached_component: Optional["RegexEntityExtractor"] = None, **kwargs: Any, ,) -> "RegexEntityExtractor"
```

Loads trained component (see parent class for full docstring).

#### persist

```python
def persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.

Return the metadata necessary to load the model again.

