---
sidebar_label: lexical_syntactic_featurizer
title: rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer
---

## LexicalSyntacticFeaturizer Objects

```python
class LexicalSyntacticFeaturizer(SparseFeaturizer)
```

Creates features for entity extraction.

Moves with a sliding window over every token in the user message and creates
features according to the configuration.

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.
Return the metadata necessary to load the model again.

