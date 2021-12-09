---
sidebar_label: rasa.nlu.training_data.entities_parser
title: rasa.nlu.training_data.entities_parser
---
## EntityAttributes Objects

```python
class EntityAttributes(NamedTuple)
```

Attributes of an entity defined in markdown data.

#### find\_entities\_in\_training\_example

```python
find_entities_in_training_example(example: Text) -> List[Dict[Text, Any]]
```

Extracts entities from an intent example.

**Arguments**:

- `example` - Intent example.
  

**Returns**:

  Extracted entities.

#### extract\_entity\_attributes

```python
extract_entity_attributes(match: Match) -> EntityAttributes
```

Extract the entity attributes, i.e. type, value, etc., from the
regex match.

**Arguments**:

- `match` - Regex match to extract the entity attributes from.
  

**Returns**:

  EntityAttributes object.

#### extract\_entity\_attributes\_from\_dict

```python
extract_entity_attributes_from_dict(entity_text: Text, match: Match) -> EntityAttributes
```

Extract entity attributes from dict format.

**Arguments**:

- `entity_text` - Original entity text.
- `match` - Regex match.
  

**Returns**:

  Extracted entity attributes.

#### get\_validated\_dict

```python
get_validated_dict(json_str: Text) -> Dict[Text, Text]
```

Converts the provided `json_str` to a valid dict containing the entity
attributes.

Users can specify entity roles, synonyms, groups for an entity in a dict, e.g.
[LA]{&quot;entity&quot;: &quot;city&quot;, &quot;role&quot;: &quot;to&quot;, &quot;value&quot;: &quot;Los Angeles&quot;}.

**Arguments**:

- `json_str` - The entity dict as string without &quot;{}&quot;.
  

**Raises**:

  ValidationError if validation of entity dict fails.
  JSONDecodeError if provided entity dict is not valid json.
  

**Returns**:

  Deserialized and validated `json_str`.

#### replace\_entities

```python
replace_entities(training_example: Text) -> Text
```

Replace special symbols related to the entities in the provided
training example.

**Arguments**:

- `training_example` - Original training example with special symbols.
  

**Returns**:

  String with removed special symbols.

#### parse\_training\_example

```python
parse_training_example(example: Text, intent: Optional[Text] = None) -> "Message"
```

Extract entities and synonyms, and convert to plain text.

