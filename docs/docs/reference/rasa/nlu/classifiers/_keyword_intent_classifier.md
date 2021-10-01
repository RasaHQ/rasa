---
sidebar_label: rasa.nlu.classifiers._keyword_intent_classifier
title: rasa.nlu.classifiers._keyword_intent_classifier
---
## KeywordIntentClassifier Objects

```python
class KeywordIntentClassifier(IntentClassifier)
```

Intent classifier using simple keyword matching.


The classifier takes a list of keywords and associated intents as an input.
An input sentence is checked for the keywords and the intent is returned.

#### process

```python
def process(message: Message, **kwargs: Any) -> None
```

Set the message intent and add it to the output is it exists.

#### persist

```python
def persist(file_name: Text, model_dir: Text) -> Dict[Text, Any]
```

Persist this model into the passed directory.

Return the metadata necessary to load the model again.

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Metadata = None, cached_component: Optional["KeywordIntentClassifier"] = None, **kwargs: Any, ,) -> "KeywordIntentClassifier"
```

Loads trained component (see parent class for full docstring).

