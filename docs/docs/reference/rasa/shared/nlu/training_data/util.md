---
sidebar_label: rasa.shared.nlu.training_data.util
title: rasa.shared.nlu.training_data.util
---
#### transform\_entity\_synonyms

```python
transform_entity_synonyms(synonyms: List[Dict[Text, Any]], known_synonyms: Optional[Dict[Text, Any]] = None) -> Dict[Text, Any]
```

Transforms the entity synonyms into a text-&gt;value dictionary

#### get\_file\_format\_extension

```python
get_file_format_extension(resource_name: Text) -> Text
```

Get the file extension based on training data format. It supports both a folder and
a file, and tries to guess the format as follows:

- if the resource is a file and has a known format, return this format&#x27;s extension
- if the resource is a folder and all the resources have the
same known format, return it&#x27;s extension
- otherwise, default to DEFAULT_FILE_FORMAT (yml).

**Arguments**:

- `resource_name` - The name of the resource, can be a file or a folder.

**Returns**:

  The resource file format.

#### remove\_untrainable\_entities\_from

```python
remove_untrainable_entities_from(example: Dict[Text, Any]) -> None
```

Remove untrainable entities from serialised training example `example`.

Entities with an untrainable extractor will be removed. Untrainable extractors
are defined in `rasa.nlu.constants.PRETRAINED_EXTRACTORS`.

**Arguments**:

- `example` - Serialised training example to inspect.

#### intent\_response\_key\_to\_template\_key

```python
intent_response_key_to_template_key(intent_response_key: Text) -> Text
```

Resolve the response template key for a given intent response key.

**Arguments**:

- `intent_response_key` - retrieval intent with the response key suffix attached.
  
- `Returns` - The corresponding response template.

#### template\_key\_to\_intent\_response\_key

```python
template_key_to_intent_response_key(template_key: Text) -> Text
```

Resolve the intent response key for the given response template.

**Arguments**:

- `template_key` - Name of the response template.
  
- `Returns` - The corresponding intent response key.

#### has\_string\_escape\_chars

```python
has_string_escape_chars(s: Text) -> bool
```

Checks whether there are any of the escape characters in the string.

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

#### sparse\_matrix\_to\_string

```python
sparse_matrix_to_string(m: scipy.sparse.spmatrix) -> Text
```

Turns a sparse matrix into a string.

Will return a line &quot;(i,j)  v&quot; for each value in the matrix.

taken from official scipy source to operate on full sparse matrix to not have
to change the `maxprint` property in-place.
https://github.com/scipy/scipy/blob/v1.7.0/scipy/sparse/base.py#L258

