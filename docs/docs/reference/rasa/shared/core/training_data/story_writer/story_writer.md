---
sidebar_label: rasa.shared.core.training_data.story_writer.story_writer
title: rasa.shared.core.training_data.story_writer.story_writer
---

## StoryWriter Objects

```python
class StoryWriter()
```

#### dumps

```python
 | @staticmethod
 | dumps(story_steps: List["StoryStep"], is_appendable: bool = False, **kwargs: Any) -> Text
```

Turns Story steps into an string.

**Arguments**:

- `story_steps` - Original story steps to be converted to the YAML.
- `is_appendable` - Specify if result should not contain
  high level keys/definitions and can be appended to
  the existing story file.

**Returns**:

  String with story steps in the desired format.

#### dump

```python
 | @staticmethod
 | dump(target: Union[Text, Path, yaml.StringIO], story_steps: List["StoryStep"], is_appendable: bool = False) -> None
```

Writes Story steps into a target file/stream.

**Arguments**:

- `target` - name of the target file/stream to write the string to.
- `story_steps` - Original story steps to be converted to the string.
- `is_appendable` - Specify if result should not contain
  high level keys/definitions and can be appended to
  the existing story file.

