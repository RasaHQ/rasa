---
sidebar_label: rasa.shared.core.training_data.story_reader.story_reader
title: rasa.shared.core.training_data.story_reader.story_reader
---
## StoryReader Objects

```python
class StoryReader()
```

Helper class to read a story file.

#### \_\_init\_\_

```python
def __init__(domain: Optional[Domain] = None, source_name: Optional[Text] = None, is_used_for_training: bool = True) -> None
```

Constructor for the StoryReader.

**Arguments**:

- `domain` - Domain object.
- `source_name` - Name of the training data source.
- `is_used_for_training` - Identifies if the user utterances should be parsed
  (entities are extracted and removed from the original text) and
  OR statements should be unfolded. This parameter is used only to
  simplify the conversation from MD story files. Don&#x27;t use it other ways,
  because it ends up in a invalid story that cannot be user for real
  training. Default value is `False`, which preserves the expected behavior
  of the reader.

#### read\_from\_file

```python
def read_from_file(filename: Text, skip_validation: bool = False) -> List[StoryStep]
```

Reads stories or rules from file.

**Arguments**:

- `filename` - Path to the story/rule file.
- `skip_validation` - `True` if file validation should be skipped.
  

**Returns**:

  `StoryStep`s read from `filename`.

#### is\_stories\_file

```python
@staticmethod
def is_stories_file(filename: Union[Text, Path]) -> bool
```

Checks if the specified file is a story file.

**Arguments**:

- `filename` - File to check.
  

**Returns**:

  `True` if specified file is a story file, `False` otherwise.

## StoryParseError Objects

```python
class StoryParseError(RasaCoreException,  ValueError)
```

Raised if there is an error while parsing a story file.

