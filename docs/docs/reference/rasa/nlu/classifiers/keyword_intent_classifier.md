---
sidebar_label: rasa.nlu.classifiers.keyword_intent_classifier
title: rasa.nlu.classifiers.keyword_intent_classifier
---
## KeywordIntentClassifierGraphComponent Objects

```python
class KeywordIntentClassifierGraphComponent(GraphComponent)
```

Intent classifier using simple keyword matching.

The classifier takes a list of keywords and associated intents as an input.
An input sentence is checked for the keywords and the intent is returned.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, intent_keyword_map: Optional[Dict] = None) -> None
```

Creates classifier.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> KeywordIntentClassifierGraphComponent
```

Creates a new untrained component (see parent class for full docstring).

#### train

```python
def train(training_data: TrainingData) -> Resource
```

Trains the intent classifier on a data set.

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Set the message intent and add it to the output if it exists.

#### persist

```python
def persist() -> None
```

Persist this model into the passed directory.

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> KeywordIntentClassifierGraphComponent
```

Loads trained component (see parent class for full docstring).

