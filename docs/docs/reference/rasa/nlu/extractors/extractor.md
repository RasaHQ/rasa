---
sidebar_label: rasa.nlu.extractors.extractor
title: rasa.nlu.extractors.extractor
---
## EntityTagSpec Objects

```python
class EntityTagSpec(NamedTuple)
```

Specification of an entity tag present in the training data.

## EntityExtractorMixin Objects

```python
class EntityExtractorMixin(abc.ABC)
```

Provides functionality for components that do entity extraction.

Inheriting from this class will add utility functions for entity extraction.
Entity extraction is the process of identifying and extracting entities like a
person&#x27;s name, or a location from a message.

#### name

```python
@property
def name() -> Text
```

Returns the name of the class.

#### add\_extractor\_name

```python
def add_extractor_name(entities: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]
```

Adds this extractor&#x27;s name to a list of entities.

**Arguments**:

- `entities` - the extracted entities.
  

**Returns**:

  the modified entities.

#### add\_processor\_name

```python
def add_processor_name(entity: Dict[Text, Any]) -> Dict[Text, Any]
```

Adds this extractor&#x27;s name to the list of processors for this entity.

**Arguments**:

- `entity` - the extracted entity and its metadata.
  

**Returns**:

  the modified entity.

#### filter\_irrelevant\_entities

```python
@staticmethod
def filter_irrelevant_entities(extracted: list, requested_dimensions: set) -> list
```

Only return dimensions the user configured.

#### filter\_trainable\_entities

```python
def filter_trainable_entities(entity_examples: List[Message]) -> List[Message]
```

Filters out untrainable entity annotations.

Creates a copy of entity_examples in which entities that have
`extractor` set to something other than
self.name (e.g. &#x27;CRFEntityExtractor&#x27;) are removed.

#### convert\_predictions\_into\_entities

```python
@staticmethod
def convert_predictions_into_entities(text: Text, tokens: List[Token], tags: Dict[Text, List[Text]], split_entities_config: Dict[Text, bool] = None, confidences: Optional[Dict[Text, List[float]]] = None) -> List[Dict[Text, Any]]
```

Convert predictions into entities.

**Arguments**:

- `text` - The text message.
- `tokens` - Message tokens without CLS token.
- `tags` - Predicted tags.
- `split_entities_config` - config for handling splitting a list of entities
- `confidences` - Confidences of predicted tags.
  

**Returns**:

  Entities.

#### get\_tag\_for

```python
@staticmethod
def get_tag_for(tags: Dict[Text, List[Text]], tag_name: Text, idx: int) -> Text
```

Get the value of the given tag name from the list of tags.

**Arguments**:

- `tags` - Mapping of tag name to list of tags;
- `tag_name` - The tag name of interest.
- `idx` - The index position of the tag.
  

**Returns**:

  The tag value.

#### check\_correct\_entity\_annotations

```python
@staticmethod
def check_correct_entity_annotations(training_data: TrainingData) -> None
```

Check if entities are correctly annotated in the training data.

If the start and end values of an entity do not match any start and end values
of the respected token, we define an entity as misaligned and log a warning.

**Arguments**:

- `training_data` - The training data.

