---
sidebar_label: rasa.shared.core.training_data.story_writer.markdown_story_writer
title: rasa.shared.core.training_data.story_writer.markdown_story_writer
---
## MarkdownStoryWriter Objects

```python
class MarkdownStoryWriter(StoryWriter)
```

Writes Core training data into a file in a markdown format.

#### dump

```python
 | @staticmethod
 | dump(target: Union[Text, Path, yaml.StringIO], story_steps: List[StoryStep], is_appendable: bool = False, is_test_story: bool = False) -> None
```

Writes Story steps into a target file/stream.

**Arguments**:

- `target` - name of the target file/stream to write the string to.
- `story_steps` - Original story steps to be converted to the string.
- `is_appendable` - Specify if result should not contain
  high level keys/definitions and can be appended to
  the existing story file.
- `is_test_story` - Identifies if the stories should be exported in test stories
  format.

#### dumps

```python
 | @staticmethod
 | dumps(story_steps: List[StoryStep], is_appendable: bool = False, is_test_story: bool = False, ignore_deprecation_warning: bool = False) -> Text
```

Turns Story steps into a markdown string.

**Arguments**:

- `story_steps` - Original story steps to be converted to the markdown.
- `is_appendable` - Specify if result should not contain
  high level keys/definitions and can be appended to
  the existing story file.
- `is_test_story` - Identifies if the stories should be exported in test stories
  format.
- `ignore_deprecation_warning` - `True` if printing the deprecation warning
  should be suppressed.
  

**Returns**:

  Story steps in the markdown format.

