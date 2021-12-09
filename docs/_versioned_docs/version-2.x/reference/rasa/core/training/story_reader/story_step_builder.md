---
sidebar_label: rasa.core.training.story_reader.story_step_builder
title: rasa.core.training.story_reader.story_step_builder
---
## StoryStepBuilder Objects

```python
class StoryStepBuilder()
```

#### add\_user\_messages

```python
 | add_user_messages(messages: List[UserUttered], unfold_or_utterances: bool = True) -> None
```

Adds next story steps with the user&#x27;s utterances.

**Arguments**:

- `messages` - User utterances.
- `unfold_or_utterances` - Identifies if the user utterance is a part of
  OR statement. This parameter is used only to simplify the conversation
  from MD story files. Don&#x27;t use it other ways, because it ends up
  in a invalid story that cannot be user for real training.
  Default value is `True`, which preserves the expected behavior
  of the reader.

