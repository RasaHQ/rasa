---
sidebar_label: rasa.shared.nlu.training_data.util
title: rasa.shared.nlu.training_data.util
---

#### transform\_entity\_synonyms

```python
transform_entity_synonyms(synonyms, known_synonyms: Optional[Dict[Text, Any]] = None) -> Dict[Text, Any]
```

Transforms the entity synonyms into a text-&gt;value dictionary

#### remove\_untrainable\_entities\_from

```python
remove_untrainable_entities_from(example: Dict[Text, Any]) -> None
```

Remove untrainable entities from serialised training example `example`.

Entities with an untrainable extractor will be removed. Untrainable extractors
are defined in `rasa.nlu.constants.PRETRAINED_EXTRACTORS`.

**Arguments**:

- `example` - Serialised training example to inspect.

#### encode\_string

```python
encode_string(s: Text) -> Text
```

Return an encoded python string.

#### decode\_string

```python
decode_string(s: Text) -> Text
```

Return a decoded python string.

#### build\_entity

```python
build_entity(start: int, end: int, value: Text, entity_type: Text, role: Optional[Text] = None, group: Optional[Text] = None, **kwargs: Any, ,) -> Dict[Text, Any]
```

Builds a standard entity dictionary.

Adds additional keyword parameters.

**Arguments**:

- `start` - start position of entity
- `end` - end position of entity
- `value` - text value of the entity
- `entity_type` - name of the entity type
- `role` - role of the entity
- `group` - group of the entity
- `**kwargs` - additional parameters
  

**Returns**:

  an entity dictionary

