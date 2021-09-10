---
sidebar_label: rasa.nlu.extractors.crf_entity_extractor
title: rasa.nlu.extractors.crf_entity_extractor
---
## CRFEntityExtractor Objects

```python
class CRFEntityExtractor(EntityExtractor)
```

#### extract\_entities

```python
def extract_entities(message: Message) -> List[Dict[Text, Any]]
```

Extract entities from the given message using the trained model(s).

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Metadata = None, cached_component: Optional["CRFEntityExtractor"] = None, **kwargs: Any, ,) -> "CRFEntityExtractor"
```

Loads trained component (see parent class for full docstring).

#### persist

```python
def persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.

Returns the metadata necessary to load the model again.

