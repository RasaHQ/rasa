---
sidebar_label: rasa.core.featurizers.tracker_featurizers
title: rasa.core.featurizers.tracker_featurizers
---
## InvalidStory Objects

```python
class InvalidStory(RasaException)
```

Exception that can be raised if story cannot be featurized.

## TrackerFeaturizer Objects

```python
class TrackerFeaturizer()
```

Base class for actual tracker featurizers.

#### \_\_init\_\_

```python
 | __init__(state_featurizer: Optional[SingleStateFeaturizer] = None) -> None
```

Initializes the tracker featurizer.

**Arguments**:

- `state_featurizer` - The state featurizer used to encode tracker states.

#### training\_states\_actions\_and\_entities

```python
 | training_states_actions_and_entities(trackers: List[DialogueStateTracker], domain: Domain, omit_unset_slots: bool = False, ignore_action_unlikely_intent: bool = False) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]
```

Transforms trackers to states, actions, and entity data.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `omit_unset_slots` - If `True` do not include the initial values of slots.
- `ignore_action_unlikely_intent` - Whether to remove `action_unlikely_intent`
  from training states.
  

**Returns**:

  Trackers as states, actions, and entity data.

#### training\_states\_and\_actions

```python
 | training_states_and_actions(trackers: List[DialogueStateTracker], domain: Domain, omit_unset_slots: bool = False, ignore_action_unlikely_intent: bool = False) -> Tuple[List[List[State]], List[List[Text]]]
```

Transforms trackers to states and actions.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `omit_unset_slots` - If `True` do not include the initial values of slots.
- `ignore_action_unlikely_intent` - Whether to remove `action_unlikely_intent`
  from training states.
  

**Returns**:

  Trackers as states and actions.

#### training\_states\_and\_labels

```python
 | training_states_and_labels(trackers: List[DialogueStateTracker], domain: Domain, omit_unset_slots: bool = False, ignore_action_unlikely_intent: bool = False) -> Tuple[List[List[State]], List[List[Text]]]
```

Transforms trackers to states and labels.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `omit_unset_slots` - If `True` do not include the initial values of slots.
- `ignore_action_unlikely_intent` - Whether to remove `action_unlikely_intent`
  from training states.
  

**Returns**:

  Trackers as states and labels.

#### training\_states\_labels\_and\_entities

```python
 | @abstractmethod
 | training_states_labels_and_entities(trackers: List[DialogueStateTracker], domain: Domain, omit_unset_slots: bool = False, ignore_action_unlikely_intent: bool = False) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]
```

Transforms trackers to states, labels, and entity data.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `omit_unset_slots` - If `True` do not include the initial values of slots.
- `ignore_action_unlikely_intent` - Whether to remove `action_unlikely_intent`
  from training states.
  

**Returns**:

  Trackers as states, labels, and entity data.

#### prepare\_for\_featurization

```python
 | prepare_for_featurization(domain: Domain, interpreter: NaturalLanguageInterpreter, bilou_tagging: bool = False) -> None
```

Ensures that the featurizer is ready to be called during training.

State featurizer needs to build its vocabulary from the domain
for it to be ready to be used during training.

**Arguments**:

- `domain` - Domain of the assistant.
- `interpreter` - NLU Interpreter for featurizing states.
- `bilou_tagging` - Whether to consider bilou tagging.

#### featurize\_trackers

```python
 | featurize_trackers(trackers: List[DialogueStateTracker], domain: Domain, interpreter: NaturalLanguageInterpreter, bilou_tagging: bool = False, ignore_action_unlikely_intent: bool = False) -> Tuple[
 |         List[List[Dict[Text, List[Features]]]],
 |         np.ndarray,
 |         List[List[Dict[Text, List[Features]]]],
 |     ]
```

Featurizes the training trackers.

**Arguments**:

- `trackers` - list of training trackers
- `domain` - the domain
- `interpreter` - the interpreter
- `bilou_tagging` - indicates whether BILOU tagging should be used or not
- `ignore_action_unlikely_intent` - Whether to remove `action_unlikely_intent`
  from training state features.
  

**Returns**:

  - a dictionary of state types (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
  ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
  turns in all training trackers
  - the label ids (e.g. action ids) for every dialogue turn in all training
  trackers
  - A dictionary of entity type (ENTITY_TAGS) to a list of features
  containing entity tag ids for text user inputs otherwise empty dict
  for all dialogue turns in all training trackers

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain, use_text_for_last_user_input: bool = False, ignore_rule_only_turns: bool = False, rule_only_data: Optional[Dict[Text, Any]] = None, ignore_action_unlikely_intent: bool = False) -> List[List[State]]
```

Transforms trackers to states for prediction.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `use_text_for_last_user_input` - Indicates whether to use text or intent label
  for featurizing last user input.
- `ignore_rule_only_turns` - If True ignore dialogue turns that are present
  only in rules.
- `rule_only_data` - Slots and loops,
  which only occur in rules but not in stories.
- `ignore_action_unlikely_intent` - Whether to remove states containing
  `action_unlikely_intent` from prediction states.
  

**Returns**:

  Trackers as states for prediction.

#### create\_state\_features

```python
 | create_state_features(trackers: List[DialogueStateTracker], domain: Domain, interpreter: NaturalLanguageInterpreter, use_text_for_last_user_input: bool = False, ignore_rule_only_turns: bool = False, rule_only_data: Optional[Dict[Text, Any]] = None, ignore_action_unlikely_intent: bool = False) -> List[List[Dict[Text, List[Features]]]]
```

Creates state features for prediction.

**Arguments**:

- `trackers` - A list of state trackers
- `domain` - The domain
- `interpreter` - The interpreter
- `use_text_for_last_user_input` - Indicates whether to use text or intent label
  for featurizing last user input.
- `ignore_rule_only_turns` - If True ignore dialogue turns that are present
  only in rules.
- `rule_only_data` - Slots and loops,
  which only occur in rules but not in stories.
- `ignore_action_unlikely_intent` - Whether to remove any states containing
  `action_unlikely_intent` from state features.
  

**Returns**:

  Dictionaries of state type (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
  ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
  turns in all trackers.

#### persist

```python
 | persist(path: Union[Text, Path]) -> None
```

Persists the tracker featurizer to the given path.

**Arguments**:

- `path` - The path to persist the tracker featurizer to.

#### load

```python
 | @staticmethod
 | load(path: Text) -> Optional["TrackerFeaturizer"]
```

Loads the featurizer from file.

**Arguments**:

- `path` - The path to load the tracker featurizer from.
  

**Returns**:

  The loaded tracker featurizer.

## FullDialogueTrackerFeaturizer Objects

```python
class FullDialogueTrackerFeaturizer(TrackerFeaturizer)
```

Creates full dialogue training data for time distributed architectures.

Creates training data that uses each time output for prediction.

#### training\_states\_labels\_and\_entities

```python
 | training_states_labels_and_entities(trackers: List[DialogueStateTracker], domain: Domain, omit_unset_slots: bool = False, ignore_action_unlikely_intent: bool = False) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]
```

Transforms trackers to states, action labels, and entity data.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `omit_unset_slots` - If `True` do not include the initial values of slots.
- `ignore_action_unlikely_intent` - Whether to remove `action_unlikely_intent`
  from training states.
  

**Returns**:

  Trackers as states, action labels, and entity data.

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain, use_text_for_last_user_input: bool = False, ignore_rule_only_turns: bool = False, rule_only_data: Optional[Dict[Text, Any]] = None, ignore_action_unlikely_intent: bool = False) -> List[List[State]]
```

Transforms trackers to states for prediction.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `use_text_for_last_user_input` - Indicates whether to use text or intent label
  for featurizing last user input.
- `ignore_rule_only_turns` - If True ignore dialogue turns that are present
  only in rules.
- `rule_only_data` - Slots and loops,
  which only occur in rules but not in stories.
- `ignore_action_unlikely_intent` - Whether to remove any states containing
  `action_unlikely_intent` from prediction states.
  

**Returns**:

  Trackers as states for prediction.

## MaxHistoryTrackerFeaturizer Objects

```python
class MaxHistoryTrackerFeaturizer(TrackerFeaturizer)
```

Truncates the tracker history into `max_history` long sequences.

Creates training data from trackers where actions are the output prediction
labels. Tracker state sequences which represent policy input are truncated
to not excede `max_history` states.

#### \_\_init\_\_

```python
 | __init__(state_featurizer: Optional[SingleStateFeaturizer] = None, max_history: Optional[int] = None, remove_duplicates: bool = True) -> None
```

Initializes the tracker featurizer.

**Arguments**:

- `state_featurizer` - The state featurizer used to encode the states.
- `max_history` - The maximum length of an extracted state sequence.
- `remove_duplicates` - Keep only unique training state sequence/label pairs.

#### slice\_state\_history

```python
 | @staticmethod
 | slice_state_history(states: List[State], slice_length: Optional[int]) -> List[State]
```

Slices states from the trackers history.

**Arguments**:

- `states` - The states
- `slice_length` - The slice length
  

**Returns**:

  The sliced states.

#### training\_states\_labels\_and\_entities

```python
 | training_states_labels_and_entities(trackers: List[DialogueStateTracker], domain: Domain, omit_unset_slots: bool = False, ignore_action_unlikely_intent: bool = False) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]
```

Transforms trackers to states, action labels, and entity data.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `omit_unset_slots` - If `True` do not include the initial values of slots.
- `ignore_action_unlikely_intent` - Whether to remove `action_unlikely_intent`
  from training states.
  

**Returns**:

  Trackers as states, labels, and entity data.

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain, use_text_for_last_user_input: bool = False, ignore_rule_only_turns: bool = False, rule_only_data: Optional[Dict[Text, Any]] = None, ignore_action_unlikely_intent: bool = False) -> List[List[State]]
```

Transforms trackers to states for prediction.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `use_text_for_last_user_input` - Indicates whether to use text or intent label
  for featurizing last user input.
- `ignore_rule_only_turns` - If True ignore dialogue turns that are present
  only in rules.
- `rule_only_data` - Slots and loops,
  which only occur in rules but not in stories.
- `ignore_action_unlikely_intent` - Whether to remove any states containing
  `action_unlikely_intent` from prediction states.
  

**Returns**:

  Trackers as states for prediction.

## IntentMaxHistoryTrackerFeaturizer Objects

```python
class IntentMaxHistoryTrackerFeaturizer(MaxHistoryTrackerFeaturizer)
```

Truncates the tracker history into `max_history` long sequences.

Creates training data from trackers where intents are the output prediction
labels. Tracker state sequences which represent policy input are truncated
to not excede `max_history` states.

#### training\_states\_labels\_and\_entities

```python
 | training_states_labels_and_entities(trackers: List[DialogueStateTracker], domain: Domain, omit_unset_slots: bool = False, ignore_action_unlikely_intent: bool = False) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]
```

Transforms trackers to states, intent labels, and entity data.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `omit_unset_slots` - If `True` do not include the initial values of slots.
- `ignore_action_unlikely_intent` - Whether to remove `action_unlikely_intent`
  from training states.
  

**Returns**:

  Trackers as states, labels, and entity data.

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain, use_text_for_last_user_input: bool = False, ignore_rule_only_turns: bool = False, rule_only_data: Optional[Dict[Text, Any]] = None, ignore_action_unlikely_intent: bool = False) -> List[List[State]]
```

Transforms trackers to states for prediction.

**Arguments**:

- `trackers` - The trackers to transform.
- `domain` - The domain.
- `use_text_for_last_user_input` - Indicates whether to use text or intent label
  for featurizing last user input.
- `ignore_rule_only_turns` - If True ignore dialogue turns that are present
  only in rules.
- `rule_only_data` - Slots and loops,
  which only occur in rules but not in stories.
- `ignore_action_unlikely_intent` - Whether to remove any states containing
  `action_unlikely_intent` from prediction states.
  

**Returns**:

  Trackers as states for prediction.

