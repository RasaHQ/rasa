---
sidebar_label: rasa.shared.core.training_data.story_reader.story_step_builder
title: rasa.shared.core.training_data.story_reader.story_step_builder
---
## StoryStepBuilder Objects

```python
class StoryStepBuilder()
```

#### add\_user\_messages

```python
 | add_user_messages(messages: List[UserUttered], is_used_for_training: bool = True) -> None
```

Adds next story steps with the user&#x27;s utterances.

**Arguments**:

- `messages` - User utterances.
- `is_used_for_training` - Identifies if the user utterance is a part of
  OR statement. This parameter is used only to simplify the conversation
  from MD story files. Don&#x27;t use it other ways, because it ends up
  in a invalid story that cannot be user for real training.
  Default value is `False`, which preserves the expected behavior
  of the reader.

