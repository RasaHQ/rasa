---
sidebar_label: rasa.shared.data
title: rasa.shared.data
---
#### yaml\_file\_extension

```python
yaml_file_extension() -> Text
```

Return YAML file extension.

#### is\_likely\_yaml\_file

```python
is_likely_yaml_file(file_path: Union[Text, Path]) -> bool
```

Check if a file likely contains yaml.

**Arguments**:

- `file_path` - path to the file
  

**Returns**:

  `True` if the file likely contains data in yaml format, `False` otherwise.

#### is\_likely\_json\_file

```python
is_likely_json_file(file_path: Text) -> bool
```

Check if a file likely contains json.

**Arguments**:

- `file_path` - path to the file
  

**Returns**:

  `True` if the file likely contains data in json format, `False` otherwise.

#### get\_core\_directory

```python
get_core_directory(paths: Optional[Union[Text, List[Text]]]) -> Text
```

Recursively collects all Core training files from a list of paths.

**Arguments**:

- `paths` - List of paths to training files or folders containing them.
  

**Returns**:

  Path to temporary directory containing all found Core training files.

#### get\_nlu\_directory

```python
get_nlu_directory(paths: Optional[Union[Text, List[Text]]]) -> Text
```

Recursively collects all NLU training files from a list of paths.

**Arguments**:

- `paths` - List of paths to training files or folders containing them.
  

**Returns**:

  Path to temporary directory containing all found NLU training files.

#### get\_data\_files

```python
get_data_files(paths: Optional[Union[Text, List[Text]]], filter_predicate: Callable[[Text], bool]) -> List[Text]
```

Recursively collects all training files from a list of paths.

**Arguments**:

- `paths` - List of paths to training files or folders containing them.
- `filter_predicate` - property to use when filtering the paths, e.g. `is_nlu_file`.
  

**Returns**:

  Paths of training data files.

#### is\_valid\_filetype

```python
is_valid_filetype(path: Union[Path, Text]) -> bool
```

Checks if given file has a supported extension.

**Arguments**:

- `path` - Path to the source file.
  

**Returns**:

  `True` is given file has supported extension, `False` otherwise.

#### is\_nlu\_file

```python
is_nlu_file(file_path: Text) -> bool
```

Checks if a file is a Rasa compatible nlu file.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a nlu file, otherwise `False`.

#### is\_config\_file

```python
is_config_file(file_path: Text) -> bool
```

Checks whether the given file path is a Rasa config file.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a Rasa config file, otherwise `False`.

## TrainingType Objects

```python
class TrainingType(Enum)
```

Enum class for defining explicitly what training types exist.

#### model\_type

```python
 | @property
 | model_type() -> Text
```

Returns the type of model which this training yields.

