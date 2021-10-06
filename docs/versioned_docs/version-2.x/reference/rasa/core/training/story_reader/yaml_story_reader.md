---
sidebar_label: rasa.core.training.story_reader.yaml_story_reader
title: rasa.core.training.story_reader.yaml_story_reader
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
 | async read_from_file(filename: Union[Text, Path]) -> List[StoryStep]
```

Read stories or rules from file.

**Arguments**:

- `filename` - Path to the story/rule file.
  

**Returns**:

  `StoryStep`s read from `filename`.

#### read\_from\_parsed\_yaml

```python
 | read_from_parsed_yaml(parsed_content: Dict[Text, Union[Dict, List]]) -> List[StoryStep]
```

Read stories from parsed YAML.

**Arguments**:

- `parsed_content` - The parsed YAML as a dictionary.
  

**Returns**:

  The parsed stories or rules.

#### is\_yaml\_story\_file

```python
 | @classmethod
 | is_yaml_story_file(cls, file_path: Text) -> bool
```

Check if file contains Core training data or rule data in YAML format.

**Arguments**:

- `file_path` - Path of the file to check.
  

**Returns**:

  `True` in case the file is a Core YAML training data or rule data file,
  `False` otherwise.

#### is\_key\_in\_yaml

```python
 | @classmethod
 | is_key_in_yaml(cls, file_path: Text, *keys: Text) -> bool
```

Check if all keys are contained in the parsed dictionary from a yaml file.

**Arguments**:

- `file_path` - path to the yaml file
- `keys` - keys to look for

**Returns**:

  `True` if all the keys are contained in the file, `False` otherwise.

#### is\_yaml\_test\_stories\_file

```python
 | @classmethod
 | is_yaml_test_stories_file(cls, file_path: Union[Text, Path]) -> bool
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

