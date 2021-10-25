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
 | __init__(domain: Optional[Domain] = None, source_name: Optional[Text] = None) -> None
```

Constructor for the StoryReader.

**Arguments**:

- `domain` - Domain object.
- `source_name` - Name of the training data source.

#### read\_from\_file

```python
 | read_from_file(filename: Text, skip_validation: bool = False) -> List[StoryStep]
```

Reads stories or rules from file.

**Arguments**:

- `filename` - Path to the story/rule file.
- `skip_validation` - `True` if file validation should be skipped.
  

**Returns**:

  `StoryStep`s read from `filename`.

#### is\_stories\_file

```python
 | @staticmethod
 | is_stories_file(filename: Union[Text, Path]) -> bool
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

