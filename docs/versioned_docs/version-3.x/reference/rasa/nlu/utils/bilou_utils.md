---
sidebar_label: rasa.nlu.utils.bilou_utils
title: rasa.nlu.utils.bilou_utils
---
#### bilou\_prefix\_from\_tag

```python
bilou_prefix_from_tag(tag: Text) -> Optional[Text]
```

Returns the BILOU prefix from the given tag.

**Arguments**:

- `tag` - the tag
  
- `Returns` - the BILOU prefix of the tag

#### tag\_without\_prefix

```python
tag_without_prefix(tag: Text) -> Text
```

Remove the BILOU prefix from the given tag.

**Arguments**:

- `tag` - the tag
  
- `Returns` - the tag without the BILOU prefix

#### bilou\_tags\_to\_ids

```python
bilou_tags_to_ids(message: "Message", tag_id_dict: Dict[Text, int], tag_name: Text = ENTITY_ATTRIBUTE_TYPE) -> List[int]
```

Maps the entity tags of the message to the ids of the provided dict.

**Arguments**:

- `message` - the message
- `tag_id_dict` - mapping of tags to ids
- `tag_name` - tag name of interest
  
- `Returns` - a list of tag ids

#### get\_bilou\_key\_for\_tag

```python
get_bilou_key_for_tag(tag_name: Text) -> Text
```

Get the message key for the BILOU tagging format of the provided tag name.

**Arguments**:

- `tag_name` - the tag name
  

**Returns**:

  the message key to store the BILOU tags

#### build\_tag\_id\_dict

```python
build_tag_id_dict(training_data: "TrainingData", tag_name: Text = ENTITY_ATTRIBUTE_TYPE) -> Optional[Dict[Text, int]]
```

Create a mapping of unique tags to ids.

**Arguments**:

- `training_data` - the training data
- `tag_name` - tag name of interest
  
- `Returns` - a mapping of tags to ids

#### apply\_bilou\_schema

```python
apply_bilou_schema(training_data: "TrainingData") -> None
```

Get a list of BILOU entity tags and set them on the given messages.

**Arguments**:

- `training_data` - the training data

#### apply\_bilou\_schema\_to\_message

```python
apply_bilou_schema_to_message(message: "Message") -> None
```

Get a list of BILOU entity tags and set them on the given message.

**Arguments**:

- `message` - the message

#### map\_message\_entities

```python
map_message_entities(message: "Message", attribute_key: Text = ENTITY_ATTRIBUTE_TYPE) -> List[Tuple[int, int, Text]]
```

Maps the entities of the given message to their start, end, and tag values.

**Arguments**:

- `message` - the message
- `attribute_key` - key of tag value to use
  
- `Returns` - a list of start, end, and tag value tuples

#### bilou\_tags\_from\_offsets

```python
bilou_tags_from_offsets(tokens: List["Token"], entities: List[Tuple[int, int, Text]]) -> List[Text]
```

Creates BILOU tags for the given tokens and entities.

**Arguments**:

- `message` - The message object.
- `tokens` - The list of tokens.
- `entities` - The list of start, end, and tag tuples.
- `missing` - The tag for missing entities.
  

**Returns**:

  BILOU tags.

#### ensure\_consistent\_bilou\_tagging

```python
ensure_consistent_bilou_tagging(predicted_tags: List[Text], predicted_confidences: List[float]) -> Tuple[List[Text], List[float]]
```

Ensure predicted tags follow the BILOU tagging schema.

We assume that starting B- tags are correct. Followed tags that belong to start
tag but have a different entity type are updated considering also the confidence
values of those tags.
For example, B-a I-b L-a is updated to B-a I-a L-a and B-a I-a O is changed to
B-a L-a.

**Arguments**:

- `predicted_tags` - predicted tags
- `predicted_confidences` - predicted confidences
  

**Returns**:

  List of tags.
  List of confidences.

