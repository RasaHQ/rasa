---
sidebar_label: rasa.shared.core.training_data.story_reader.yaml_story_reader
title: rasa.shared.core.training_data.story_reader.yaml_story_reader
---
## YAMLStoryReader Objects

```python
class YAMLStoryReader(StoryReader)
```

Class that reads Core training data and rule data in YAML format.

#### from\_reader

```python
@classmethod
def from_reader(cls, reader: "YAMLStoryReader") -> "YAMLStoryReader"
```

Create a reader from another reader.

**Arguments**:

- `reader` - Another reader.
  

**Returns**:

  A new reader instance.

#### read\_from\_file

```python
def read_from_file(filename: Union[Text, Path], skip_validation: bool = False) -> List[StoryStep]
```

Read stories or rules from file.

**Arguments**:

- `filename` - Path to the story/rule file.
- `skip_validation` - `True` if the file was already validated
  e.g. when it was stored in the database.
  

**Returns**:

  `StoryStep`s read from `filename`.

#### read\_from\_string

```python
def read_from_string(string: Text, skip_validation: bool = False) -> List[StoryStep]
```

Read stories or rules from a string.

**Arguments**:

- `string` - Unprocessed YAML file content.
- `skip_validation` - `True` if the string was already validated
  e.g. when it was stored in the database.
  

**Returns**:

  `StoryStep`s read from `string`.

#### read\_from\_parsed\_yaml

```python
def read_from_parsed_yaml(parsed_content: Dict[Text, Union[Dict, List]]) -> List[StoryStep]
```

Read stories from parsed YAML.

**Arguments**:

- `parsed_content` - The parsed YAML as a dictionary.
  

**Returns**:

  The parsed stories or rules.

#### is\_stories\_file

```python
@classmethod
def is_stories_file(cls, file_path: Union[Text, Path]) -> bool
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

#### is\_test\_stories\_file

```python
@classmethod
def is_test_stories_file(cls, file_path: Union[Text, Path]) -> bool
```

Checks if a file is a test conversations file.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a conversation test file, otherwise `False`.

#### unpack\_regex\_message

```python
@staticmethod
def unpack_regex_message(message: Message, domain: Optional[Domain] = None, entity_extractor_name: Optional[Text] = None) -> Message
```

Unpacks the message if `TEXT` contains an encoding of attributes.

**Arguments**:

- `message` - some message
- `domain` - the domain
- `entity_extractor_name` - An extractor name which should be added for the
  entities.
  

**Returns**:

  the given message if that message does not need to be unpacked, and a new
  message with the extracted attributes otherwise

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

