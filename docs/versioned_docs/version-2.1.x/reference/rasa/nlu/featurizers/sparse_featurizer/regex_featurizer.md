---
sidebar_label: regex_featurizer
title: rasa.nlu.featurizers.sparse_featurizer.regex_featurizer
---

## RegexFeaturizer Objects

```python
class RegexFeaturizer(SparseFeaturizer)
```

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.
Return the metadata necessary to load the model again.

