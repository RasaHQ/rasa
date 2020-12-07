---
sidebar_label: regex_featurizer
title: rasa.nlu.featurizers.sparse_featurizer.regex_featurizer
---

## RegexFeaturizer Objects

```python
class RegexFeaturizer(SparseFeaturizer)
```

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None, known_patterns: Optional[List[Dict[Text, Text]]] = None) -> None
```

Construct new features for regexes and lookup table using regex expressions.

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.
Return the metadata necessary to load the model again.

