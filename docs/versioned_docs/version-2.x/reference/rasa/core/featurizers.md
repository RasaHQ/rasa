---
sidebar_label: rasa.core.featurizers
title: rasa.core.featurizers
---
## SingleStateFeaturizer Objects

```python
class SingleStateFeaturizer()
```

Base class for mechanisms to transform the conversations state into ML formats.

Subclasses of SingleStateFeaturizer decide how the bot will transform
the conversation state to a format which a classifier can read:
feature vector.

#### prepare\_from\_domain

```python
 | prepare_from_domain(domain: Domain) -> None
```

Helper method to init based on domain.

#### encode

```python
 | encode(state: Dict[Text, float]) -> np.ndarray
```

Encode user input.

#### action\_as\_one\_hot

```python
 | @staticmethod
 | action_as_one_hot(action: Text, domain: Domain) -> np.ndarray
```

Encode system action as one-hot vector.

#### create\_encoded\_all\_actions

```python
 | create_encoded_all_actions(domain: Domain) -> np.ndarray
```

Create matrix with all actions from domain encoded in rows.

## BinarySingleStateFeaturizer Objects

```python
class BinarySingleStateFeaturizer(SingleStateFeaturizer)
```

Assumes all features are binary.

All features should be either on or off, denoting them with 1 or 0.

#### \_\_init\_\_

```python
 | __init__() -> None
```

Declares instant variables.

#### prepare\_from\_domain

```python
 | prepare_from_domain(domain: Domain) -> None
```

Use Domain to prepare featurizer.

#### encode

```python
 | encode(state: Dict[Text, float]) -> np.ndarray
```

Returns a binary vector indicating which features are active.

Given a dictionary of states (e.g. &#x27;intent_greet&#x27;,
&#x27;prev_action_listen&#x27;,...) return a binary vector indicating which
features of `self.input_features` are in the bag. NB it&#x27;s a
regular double precision float array type.

For example with two active features out of five possible features
this would return a vector like `[0 0 1 0 1]`

If intent features are given with a probability, for example
with two active features and two uncertain intents out
of five possible features this would return a vector
like `[0.3, 0.7, 1.0, 0, 1.0]`.

If this is just a padding vector we set all values to `-1`.
padding vectors are specified by a `None` or `[None]`
value for states.

#### create\_encoded\_all\_actions

```python
 | create_encoded_all_actions(domain: Domain) -> np.ndarray
```

Create matrix with all actions from domain encoded in rows as bag of words

## LabelTokenizerSingleStateFeaturizer Objects

```python
class LabelTokenizerSingleStateFeaturizer(SingleStateFeaturizer)
```

Creates bag-of-words feature vectors.

User intents and bot action names are split into tokens
and used to create bag-of-words feature vectors.

**Arguments**:

- `split_symbol` - The symbol that separates words in
  intets and action names.
  
- `use_shared_vocab` - The flag that specifies if to create
  the same vocabulary for user intents and bot actions.

#### \_\_init\_\_

```python
 | __init__(use_shared_vocab: bool = False, split_symbol: Text = "_") -> None
```

inits vocabulary for label bag of words representation

#### prepare\_from\_domain

```python
 | prepare_from_domain(domain: Domain) -> None
```

Creates internal vocabularies for user intents and bot actions.

#### encode

```python
 | encode(state: Dict[Text, float]) -> np.ndarray
```

Returns a binary vector indicating which tokens are present.

#### create\_encoded\_all\_actions

```python
 | create_encoded_all_actions(domain: Domain) -> np.ndarray
```

Create matrix with all actions from domain encoded in rows as bag of words

## TrackerFeaturizer Objects

```python
class TrackerFeaturizer()
```

Base class for actual tracker featurizers.

#### training\_states\_and\_actions

```python
 | training_states_and_actions(trackers: List[DialogueStateTracker], domain: Domain) -> Tuple[List[List[Dict]], List[List[Text]]]
```

Transforms list of trackers to lists of states and actions.

#### featurize\_trackers

```python
 | featurize_trackers(trackers: List[DialogueStateTracker], domain: Domain) -> DialogueTrainingData
```

Create training data.

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain) -> List[List[Dict[Text, float]]]
```

Transforms list of trackers to lists of states for prediction.

#### create\_X

```python
 | create_X(trackers: List[DialogueStateTracker], domain: Domain) -> np.ndarray
```

Create X for prediction.

#### load

```python
 | @staticmethod
 | load(path) -> Optional["TrackerFeaturizer"]
```

Loads the featurizer from file.

## FullDialogueTrackerFeaturizer Objects

```python
class FullDialogueTrackerFeaturizer(TrackerFeaturizer)
```

Creates full dialogue training data for time distributed architectures.

Creates training data that uses each time output for prediction.
Training data is padded up to the length of the longest dialogue with -1.

#### training\_states\_and\_actions

```python
 | training_states_and_actions(trackers: List[DialogueStateTracker], domain: Domain) -> Tuple[List[List[Dict]], List[List[Text]]]
```

Transforms list of trackers to lists of states and actions.

Training data is padded up to the length of the longest dialogue with -1.

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain) -> List[List[Dict[Text, float]]]
```

Transforms list of trackers to lists of states for prediction.

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
 | slice_state_history(states: List[Dict[Text, float]], slice_length: Optional[int]) -> List[Optional[Dict[Text, float]]]
```

Slices states from the trackers history.

If the slice is at the array borders, padding will be added to ensure
the slice length.

#### training\_states\_and\_actions

```python
 | training_states_and_actions(trackers: List[DialogueStateTracker], domain: Domain) -> Tuple[List[List[Optional[Dict[Text, float]]]], List[List[Text]]]
```

Transforms list of trackers to lists of states and actions.

Training data is padded up to the max_history with -1.

#### prediction\_states

```python
 | prediction_states(trackers: List[DialogueStateTracker], domain: Domain) -> List[List[Dict[Text, float]]]
```

Transforms list of trackers to lists of states for prediction.

