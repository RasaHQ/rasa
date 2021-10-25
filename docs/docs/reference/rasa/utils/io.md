---
sidebar_label: rasa.utils.io
title: rasa.utils.io
---
#### pickle\_dump

```python
pickle_dump(filename: Union[Text, Path], obj: Any) -> None
```

Saves object to file.

**Arguments**:

- `filename` - the filename to save the object to
- `obj` - the object to store

#### pickle\_load

```python
pickle_load(filename: Union[Text, Path]) -> Any
```

Loads an object from a file.

**Arguments**:

- `filename` - the filename to load the object from
  
- `Returns` - the loaded object

#### create\_temporary\_file

```python
create_temporary_file(data: Any, suffix: Text = "", mode: Text = "w+") -> Text
```

Creates a tempfile.NamedTemporaryFile object for data.

#### create\_temporary\_directory

```python
create_temporary_directory() -> Text
```

Creates a tempfile.TemporaryDirectory.

#### create\_path

```python
create_path(file_path: Text) -> None
```

Makes sure all directories in the &#x27;file_path&#x27; exists.

#### file\_type\_validator

```python
file_type_validator(valid_file_types: List[Text], error_message: Text) -> Type["Validator"]
```

Creates a `Validator` class which can be used with `questionary` to validate
file paths.

#### not\_empty\_validator

```python
not_empty_validator(error_message: Text) -> Type["Validator"]
```

Creates a `Validator` class which can be used with `questionary` to validate
that the user entered something other than whitespace.

#### create\_validator

```python
create_validator(function: Callable[[Text], bool], error_message: Text) -> Type["Validator"]
```

Helper method to create `Validator` classes from callable functions. Should be
removed when questionary supports `Validator` objects.

#### json\_unpickle

```python
json_unpickle(file_name: Union[Text, Path], encode_non_string_keys: bool = False) -> Any
```

Unpickle an object from file using json.

**Arguments**:

- `file_name` - the file to load the object from
- `encode_non_string_keys` - If set to `True` then jsonpickle will encode non-string
  dictionary keys instead of coercing them into strings via `repr()`.
  
- `Returns` - the object

#### json\_pickle

```python
json_pickle(file_name: Union[Text, Path], obj: Any, encode_non_string_keys: bool = False) -> None
```

Pickle an object to a file using json.

**Arguments**:

- `file_name` - the file to store the object to
- `obj` - the object to store
- `encode_non_string_keys` - If set to `True` then jsonpickle will encode non-string
  dictionary keys instead of coercing them into strings via `repr()`.

#### get\_emoji\_regex

```python
get_emoji_regex() -> Pattern
```

Returns regex to identify emojis.

#### are\_directories\_equal

```python
are_directories_equal(dir1: Path, dir2: Path) -> bool
```

Compares two directories recursively.

Files in each directory are
assumed to be equal if their names and contents are equal.

**Arguments**:

- `dir1` - The first directory.
- `dir2` - The second directory.
  

**Returns**:

  `True` if they are equal, `False` otherwise.

