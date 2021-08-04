---
sidebar_label: keyword_intent_classifier
title: rasa.nlu.classifiers.keyword_intent_classifier
---

## KeywordIntentClassifier Objects

```python
class KeywordIntentClassifier(IntentClassifier)
```

Intent classifier using simple keyword matching.


The classifier takes a list of keywords and associated intents as an input.
An input sentence is checked for the keywords and the intent is returned.

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Dict[Text, Any]
```

Persist this model into the passed directory.

Return the metadata necessary to load the model again.

