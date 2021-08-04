---
sidebar_label: crf_entity_extractor
title: rasa.nlu.extractors.crf_entity_extractor
---

## CRFEntityExtractor Objects

```python
class CRFEntityExtractor(EntityExtractor)
```

#### extract\_entities

```python
 | extract_entities(message: Message) -> List[Dict[Text, Any]]
```

Extract entities from the given message using the trained model(s).

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.

Returns the metadata necessary to load the model again.

