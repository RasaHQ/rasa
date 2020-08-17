---
sidebar_label: rasa.nlu.utils
title: rasa.nlu.utils
---

#### relative\_normpath

```python
relative_normpath(f: Optional[Text], path: Text) -> Optional[Path]
```

Return the path of file relative to `path`.

#### module\_path\_from\_object

```python
module_path_from_object(o: Any) -> Text
```

Returns the fully qualified class path of the instantiated object.

#### write\_json\_to\_file

```python
write_json_to_file(filename: Text, obj: Any, **kwargs: Any) -> None
```

Write an object as a json string to a file.

#### write\_to\_file

```python
write_to_file(filename: Text, text: Any) -> None
```

Write a text to a file.

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

#### is\_model\_dir

```python
is_model_dir(model_dir: Text) -> bool
```

Checks if the given directory contains a model and can be safely removed.

specifically checks if the directory has no subdirectories and
if all files have an appropriate ending.

#### is\_url

```python
is_url(resource_name: Text) -> bool
```

Return True if string is an http, ftp, or file URL path.

This implementation is the same as the one used by matplotlib

#### remove\_model

```python
remove_model(model_dir: Text) -> bool
```

Removes a model directory and all its content.

