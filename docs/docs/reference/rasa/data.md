---
sidebar_label: rasa.data
title: rasa.data
---

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

#### get\_core\_nlu\_files

```python
get_core_nlu_files(paths: Optional[Union[Text, List[Text]]]) -> Tuple[List[Text], List[Text]]
```

Recursively collects all training files from a list of paths.

**Arguments**:

- `paths` - List of paths to training files or folders containing them.
  

**Returns**:

  Tuple of paths to story and NLU files.

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

#### is\_end\_to\_end\_conversation\_test\_file

```python
is_end_to_end_conversation_test_file(file_path: Text) -> bool
```

Checks if a file is an end-to-end conversation test file.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a conversation test file, otherwise `False`.

#### is\_config\_file

```python
is_config_file(file_path: Text) -> bool
```

Checks whether the given file path is a Rasa config file.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a Rasa config file, otherwise `False`.

