---
sidebar_label: regex_entity_extractor
title: rasa.nlu.extractors.regex_entity_extractor
---

## RegexEntityExtractor Objects

```python
class RegexEntityExtractor(EntityExtractor)
```

Searches for entities in the user&#x27;s message using the lookup tables and regexes
defined in the training data.

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None, patterns: Optional[List[Dict[Text, Text]]] = None)
```

Extracts entities using the lookup tables and/or regexes defined.

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.
Return the metadata necessary to load the model again.

