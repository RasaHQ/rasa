---
sidebar_label: rasa.data
title: rasa.data
---
#### get\_test\_directory

```python
get_test_directory(paths: Optional[Union[Text, List[Text]]]) -> Text
```

Recursively collects all Core training files from a list of paths.

**Arguments**:

- `paths` - List of paths to training files or folders containing them.
  

**Returns**:

  Path to temporary directory containing all found Core training files.

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

#### get\_core\_nlu\_directories

```python
get_core_nlu_directories(paths: Optional[Union[Text, List[Text]]]) -> Tuple[Text, Text]
```

Recursively collects all training files from a list of paths.

**Arguments**:

- `paths` - List of paths to training files or folders containing them.
  

**Returns**:

  Path to directory containing the Core files and path to directory
  containing the NLU training files.

#### get\_data\_files

```python
get_data_files(paths: Optional[Union[Text, List[Text]]], filter_predicate: Callable[[Text], bool]) -> List[Text]
```

Recursively collects all training files from a list of paths.

**Arguments**:

- `paths` - List of paths to training files or folders containing them.
- `filter_predicate` - property to use when filtering the paths, e.g. `is_nlu_file`.
  

**Returns**:

  paths of training data files.

#### is\_valid\_filetype

```python
is_valid_filetype(path: Union[Text, Path]) -> bool
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

#### is\_story\_file

```python
is_story_file(file_path: Text) -> bool
```

Checks if a file is a Rasa story file.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a story file, otherwise `False`.

#### is\_test\_stories\_file

```python
is_test_stories_file(file_path: Text) -> bool
```

Checks if a file is a test stories file.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a story file containing tests, otherwise `False`.

#### is\_config\_file

```python
is_config_file(file_path: Text) -> bool
```

Checks whether the given file path is a Rasa config file.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a Rasa config file, otherwise `False`.

