---
sidebar_label: tracker_featurizers
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

Initialize the tracker featurizer.

**Arguments**:

- `state_featurizer` - The state featurizer used to encode the states.

#### training\_states\_and\_actions

```python
 | training_states_and_actions(trackers: List[DialogueStateTracker], domain: Domain) -> Tuple[List[List[State]], List[List[Text]]]
```

Transforms list of trackers to lists of states and actions.

**Arguments**:

- `trackers` - The trackers to transform
- `domain` - The domain
  

**Returns**:

  A tuple of list of states and list of actions.

#### featurize\_trackers

```python
 | featurize_trackers(trackers: List[DialogueStateTracker], domain: Domain, interpreter: NaturalLanguageInterpreter) -> Tuple[List[List[Dict[Text, List["Features"]]]], np.ndarray]
```

Featurize the training trackers.

**Arguments**:

- `trackers` - list of training trackers
- `domain` - the domain
- `interpreter` - the interpreter
  

**Returns**:

  - a dictionary of state types (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
  ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
  turns in all training trackers
  - the label ids (e.g. action ids) for every dialuge turn in all training
  trackers

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain) -> List[List[State]]
```

Transforms list of trackers to lists of states for prediction.

**Arguments**:

- `trackers` - The trackers to transform
- `domain` - The domain
  

**Returns**:

  A list of states.

#### create\_state\_features

```python
 | create_state_features(trackers: List[DialogueStateTracker], domain: Domain, interpreter: NaturalLanguageInterpreter) -> List[List[Dict[Text, List["Features"]]]]
```

Create state features for prediction.

**Arguments**:

- `trackers` - A list of state trackers
- `domain` - The domain
- `interpreter` - The interpreter
  

**Returns**:

  A dictionary of state type (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
  ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
  turns in all trackers.

#### persist

```python
 | persist(path: Union[Text, Path]) -> None
```

Persist the tracker featurizer to the given path.

**Arguments**:

- `path` - The path to persist the tracker featurizer to.

#### load

```python
 | @staticmethod
 | load(path: Text) -> Optional["TrackerFeaturizer"]
```

Load the featurizer from file.

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
Training data is padded up to the length of the longest dialogue with -1.

#### training\_states\_and\_actions

```python
 | training_states_and_actions(trackers: List[DialogueStateTracker], domain: Domain) -> Tuple[List[List[State]], List[List[Text]]]
```

Transforms list of trackers to lists of states and actions.

Training data is padded up to the length of the longest dialogue with -1.

**Arguments**:

- `trackers` - The trackers to transform
- `domain` - The domain
  

**Returns**:

  A tuple of list of states and list of actions.

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain) -> List[List[State]]
```

Transforms list of trackers to lists of states for prediction.

**Arguments**:

- `trackers` - The trackers to transform
- `domain` - The domain
  

**Returns**:

  A list of states.

## MaxHistoryTrackerFeaturizer Objects

```python
class MaxHistoryTrackerFeaturizer(TrackerFeaturizer)
```

Slices the tracker history into max_history batches.

Creates training data that uses last output for prediction.
Training data is padded up to the max_history with -1.

#### slice\_state\_history

```python
 | @staticmethod
 | slice_state_history(states: List[State], slice_length: Optional[int]) -> List[State]
```

Slice states from the trackers history.

If the slice is at the array borders, padding will be added to ensure
the slice length.

**Arguments**:

- `states` - The states
- `slice_length` - The slice length
  

**Returns**:

  The sliced states.

#### training\_states\_and\_actions

```python
 | training_states_and_actions(trackers: List[DialogueStateTracker], domain: Domain) -> Tuple[List[List[State]], List[List[Text]]]
```

Transforms list of trackers to lists of states and actions.

Training data is padded up to the length of the longest dialogue with -1.

**Arguments**:

- `trackers` - The trackers to transform
- `domain` - The domain
  

**Returns**:

  A tuple of list of states and list of actions.

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain) -> List[List[State]]
```

Transforms list of trackers to lists of states for prediction.

**Arguments**:

- `trackers` - The trackers to transform
- `domain` - The domain
  

**Returns**:

  A list of states.

