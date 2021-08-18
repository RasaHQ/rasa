---
sidebar_label: yaml_story_reader
title: rasa.shared.core.training_data.story_reader.yaml_story_reader
---

## YAMLStoryReader Objects

```python
class YAMLStoryReader(StoryReader)
```

Class that reads Core training data and rule data in YAML format.

#### from\_reader

```python
 | @classmethod
 | from_reader(cls, reader: "YAMLStoryReader") -> "YAMLStoryReader"
```

Create a reader from another reader.

**Arguments**:

- `reader` - Another reader.
  

**Returns**:

  A new reader instance.

#### read\_from\_file

```python
 | read_from_file(filename: Union[Text, Path]) -> List[StoryStep]
```

Read stories or rules from file.

**Arguments**:

- `filename` - Path to the story/rule file.
  

**Returns**:

  `StoryStep`s read from `filename`.

#### read\_from\_string

```python
 | read_from_string(string: Text) -> List[StoryStep]
```

Read stories or rules from a string.

**Arguments**:

- `string` - Unprocessed YAML file content.
  

**Returns**:

  `StoryStep`s read from `string`.

#### read\_from\_parsed\_yaml

```python
 | read_from_parsed_yaml(parsed_content: Dict[Text, Union[Dict, List]]) -> List[StoryStep]
```

Read stories from parsed YAML.

**Arguments**:

- `parsed_content` - The parsed YAML as a dictionary.
  

**Returns**:

  The parsed stories or rules.

#### is\_stories\_file

```python
 | @classmethod
 | is_stories_file(cls, file_path: Union[Text, Path]) -> bool
```

Check if file contains Core training data or rule data in YAML format.

**Arguments**:

- `file_path` - Path of the file to check.
  

**Returns**:

  `True` in case the file is a Core YAML training data or rule data file,
  `False` otherwise.
  

**Raises**:

- `YamlException` - if the file seems to be a YAML file (extension) but
  can not be read / parsed.

#### is\_key\_in\_yaml

```python
 | @classmethod
 | is_key_in_yaml(cls, file_path: Union[Text, Path], *keys: Text) -> bool
```

Check if any of the keys is contained in the root object of the yaml file.

**Arguments**:

- `file_path` - path to the yaml file
- `keys` - keys to look for
  

**Returns**:

  `True` if at least one of the keys is found, `False` otherwise.
  

**Raises**:

- `FileNotFoundException` - if the file cannot be found.

#### is\_test\_stories\_file

```python
 | @classmethod
 | is_test_stories_file(cls, file_path: Union[Text, Path]) -> bool
```

Checks if a file is a test conversations file.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a conversation test file, otherwise `False`.

## StoryParser Objects

```python
class StoryParser(YAMLStoryReader)
```

Encapsulate story-specific parser behavior.

## RuleParser Objects

```python
class RuleParser(YAMLStoryReader)
```

Encapsulate rule-specific parser behavior.

