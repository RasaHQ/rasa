---
sidebar_label: rasa.core.training.story_reader.story_reader
title: rasa.core.training.story_reader.story_reader
---
## StoryReader Objects

```python
class StoryReader()
```

Helper class to read a story file.

#### \_\_init\_\_

```python
 | __init__(domain: Optional[Domain] = None, template_vars: Optional[Dict] = None, use_e2e: bool = False, source_name: Text = None, unfold_or_utterances: bool = True) -> None
```

Constructor for the StoryReader.

**Arguments**:

- `domain` - Domain object.
- `template_vars` - Template variables to be replaced.
- `use_e2e` - Specifies whether to use the e2e parser or not.
- `source_name` - Name of the training data source.
- `unfold_or_utterances` - Identifies if the user utterance is a part of
  OR statement. This parameter is used only to simplify the conversation
  from MD story files. Don&#x27;t use it other ways, because it ends up
  in a invalid story that cannot be user for real training.
  Default value is `True`, which preserves the expected behavior
  of the reader.

