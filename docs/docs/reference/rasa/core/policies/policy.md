---
sidebar_label: rasa.core.policies.policy
title: rasa.core.policies.policy
---

## SupportedData Objects

```python
class SupportedData(Enum)
```

Enumeration of a policy&#x27;s supported training data type.

#### trackers\_for\_policy

```python
 | @staticmethod
 | trackers_for_policy(policy: Union["Policy", Type["Policy"]], trackers: Union[List[DialogueStateTracker], List[TrackerWithCachedStates]]) -> Union[List[DialogueStateTracker], List[TrackerWithCachedStates]]
```

Return trackers for a given policy.

**Arguments**:

- `policy` - Policy or policy type to return trackers for.
- `trackers` - Trackers to split.
  

**Returns**:

  Trackers from ML-based training data and/or rule-based data.

## Policy Objects

```python
class Policy()
```

#### supported\_data

```python
 | @staticmethod
 | supported_data() -> SupportedData
```

The type of data supported by this policy.

By default, this is only ML-based training data. If policies support rule data,
or both ML-based data and rule data, they need to override this method.

**Returns**:

  The data type supported by this policy (ML-based training data).

#### featurize\_for\_training

```python
 | featurize_for_training(training_trackers: List[DialogueStateTracker], domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> Tuple[List[List[Dict[Text, List["Features"]]]], np.ndarray]
```

Transform training trackers into a vector representation.

The trackers, consisting of multiple turns, will be transformed
into a float vector which can be used by a ML model.

**Arguments**:

  training_trackers:
  the list of the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `interpreter` - the :class:`rasa.core.interpreter.NaturalLanguageInterpreter`
  

**Returns**:

  - a dictionary of attribute (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
  ENTITIES, SLOTS, FORM) to a list of features for all dialogue turns in
  all training trackers
  - the label ids (e.g. action ids) for every dialogue turn in all training
  trackers

#### train

```python
 | train(training_trackers: List[TrackerWithCachedStates], domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> None
```

Trains the policy on given training trackers.

**Arguments**:

  training_trackers:
  the list of the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `interpreter` - Interpreter which can be used by the polices for featurization.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> List[float]
```

Predicts the next action the bot should take after seeing the tracker.

**Arguments**:

- `tracker` - the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `interpreter` - Interpreter which may be used by the policies to create
  additional features.
  

**Returns**:

  the list of probabilities for the next actions

#### persist

```python
 | persist(path: Union[Text, Path]) -> None
```

Persists the policy to storage.

**Arguments**:

- `path` - Path to persist policy to.

#### load

```python
 | @classmethod
 | load(cls, path: Union[Text, Path]) -> "Policy"
```

Loads a policy from path.

**Arguments**:

- `path` - Path to load policy from.
  

**Returns**:

  An instance of `Policy`.

#### format\_tracker\_states

```python
 | format_tracker_states(states: List[Dict]) -> Text
```

Format tracker states to human readable format on debug log.

**Arguments**:

- `states` - list of tracker states dicts
  

**Returns**:

  the string of the states with user intents and actions

#### confidence\_scores\_for

```python
confidence_scores_for(action_name: Text, value: float, domain: Domain) -> List[float]
```

Returns confidence scores if a single action is predicted.

**Arguments**:

- `action_name` - the name of the action for which the score should be set
- `value` - the confidence for `action_name`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
  

**Returns**:

  the list of the length of the number of actions

