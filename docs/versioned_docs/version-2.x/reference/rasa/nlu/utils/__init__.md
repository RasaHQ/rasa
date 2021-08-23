---
sidebar_label: rasa.nlu.utils
title: rasa.nlu.utils
---
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

#### is\_url

```python
is_url(resource_name: Text) -> bool
```

Check whether the url specified is a well formed one.

**Arguments**:

- `resource_name` - Remote URL to validate
  

**Returns**:

  `True` if valid, otherwise `False`.

