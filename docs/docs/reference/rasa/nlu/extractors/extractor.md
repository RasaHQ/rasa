---
sidebar_label: extractor
title: rasa.nlu.extractors.extractor
---

## EntityExtractor Objects

```python
class EntityExtractor(Component)
```

#### init\_split\_entities

```python
 | init_split_entities()
```

Initialise the behaviour for splitting entities by comma (or not).

#### filter\_irrelevant\_entities

```python
 | @staticmethod
 | filter_irrelevant_entities(extracted: list, requested_dimensions: set) -> list
```

Only return dimensions the user configured.

#### filter\_trainable\_entities

```python
 | filter_trainable_entities(entity_examples: List[Message]) -> List[Message]
```

Filters out untrainable entity annotations.

Creates a copy of entity_examples in which entities that have
`extractor` set to something other than
self.name (e.g. &#x27;CRFEntityExtractor&#x27;) are removed.

#### convert\_predictions\_into\_entities

```python
 | @staticmethod
 | convert_predictions_into_entities(text: Text, tokens: List[Token], tags: Dict[Text, List[Text]], split_entities_config: Dict[Text, bool] = None, confidences: Optional[Dict[Text, List[float]]] = None) -> List[Dict[Text, Any]]
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
 | @staticmethod
 | get_tag_for(tags: Dict[Text, List[Text]], tag_name: Text, idx: int) -> Text
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
 | @staticmethod
 | check_correct_entity_annotations(training_data: TrainingData) -> None
```

Check if entities are correctly annotated in the training data.

If the start and end values of an entity do not match any start and end values
of the respected token, we define an entity as misaligned and log a warning.

**Arguments**:

- `training_data` - The training data.

