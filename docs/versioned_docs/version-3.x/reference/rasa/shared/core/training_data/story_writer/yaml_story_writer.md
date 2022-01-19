---
sidebar_label: rasa.shared.core.training_data.story_writer.yaml_story_writer
title: rasa.shared.core.training_data.story_writer.yaml_story_writer
---
## YAMLStoryWriter Objects

```python
class YAMLStoryWriter(StoryWriter)
```

Writes Core training data into a file in a YAML format.

#### dumps

```python
 | dumps(story_steps: List[StoryStep], is_appendable: bool = False, is_test_story: bool = False) -> Text
```

Turns Story steps into an YAML string.

**Arguments**:

- `story_steps` - Original story steps to be converted to the YAML.
- `is_appendable` - Specify if result should not contain
  high level keys/definitions and can be appended to
  the existing story file.
- `is_test_story` - Identifies if the stories should be exported in test stories
  format.
  

**Returns**:

  String with story steps in the YAML format.

#### dump

```python
 | dump(target: Union[Text, Path, yaml.StringIO], story_steps: List[StoryStep], is_appendable: bool = False, is_test_story: bool = False) -> None
```

Writes Story steps into a target file/stream.

**Arguments**:

- `target` - name of the target file/stream to write the YAML to.
- `story_steps` - Original story steps to be converted to the YAML.
- `is_appendable` - Specify if result should not contain
  high level keys/definitions and can be appended to
  the existing story file.
- `is_test_story` - Identifies if the stories should be exported in test stories
  format.

#### stories\_to\_yaml

```python
 | stories_to_yaml(story_steps: List[StoryStep], is_test_story: bool = False) -> Dict[Text, Any]
```

Converts a sequence of story steps into yaml format.

**Arguments**:

- `story_steps` - Original story steps to be converted to the YAML.
- `is_test_story` - `True` if the story is an end-to-end conversation test story.

#### process\_story\_step

```python
 | process_story_step(story_step: StoryStep) -> OrderedDict
```

Converts a single story step into an ordered dict.

**Arguments**:

- `story_step` - A single story step to be converted to the dict.
  

**Returns**:

  Dict with a story step.

#### stories\_contain\_loops

```python
 | @staticmethod
 | stories_contain_loops(stories: List[StoryStep]) -> bool
```

Checks if the stories contain at least one active loop.

**Arguments**:

- `stories` - Stories steps.
  

**Returns**:

  `True` if the `stories` contain at least one active loop.
  `False` otherwise.

#### process\_user\_utterance

```python
 | @staticmethod
 | process_user_utterance(user_utterance: UserUttered, is_test_story: bool = False) -> OrderedDict
```

Converts a single user utterance into an ordered dict.

**Arguments**:

- `user_utterance` - Original user utterance object.
- `is_test_story` - Identifies if the user utterance should be added
  to the final YAML or not.
  

**Returns**:

  Dict with a user utterance.

#### process\_action

```python
 | @staticmethod
 | process_action(action: ActionExecuted) -> Optional[OrderedDict]
```

Converts a single action into an ordered dict.

**Arguments**:

- `action` - Original action object.
  

**Returns**:

  Dict with an action.

#### process\_slot

```python
 | @staticmethod
 | process_slot(event: SlotSet) -> OrderedDict
```

Converts a single `SlotSet` event into an ordered dict.

**Arguments**:

- `event` - Original `SlotSet` event.
  

**Returns**:

  OrderedDict with an `SlotSet` event.

#### process\_checkpoints

```python
 | @staticmethod
 | process_checkpoints(checkpoints: List[Checkpoint]) -> List[OrderedDict]
```

Converts checkpoints event into an ordered dict.

**Arguments**:

- `checkpoints` - List of original checkpoint.
  

**Returns**:

  List of converted checkpoints.

#### process\_or\_utterances

```python
 | process_or_utterances(utterances: List[UserUttered]) -> OrderedDict
```

Converts user utterance containing the `OR` statement.

**Arguments**:

- `utterances` - User utterances belonging to the same `OR` statement.
  

**Returns**:

  Dict with converted user utterances.

#### process\_active\_loop

```python
 | @staticmethod
 | process_active_loop(event: ActiveLoop) -> OrderedDict
```

Converts ActiveLoop event into an ordered dict.

**Arguments**:

- `event` - ActiveLoop event.
  

**Returns**:

  Converted event.

#### process\_rule\_step

```python
 | process_rule_step(rule_step: RuleStep) -> OrderedDict
```

Converts a RuleStep into an ordered dict.

**Arguments**:

- `rule_step` - RuleStep object.
  

**Returns**:

  Converted rule step.

