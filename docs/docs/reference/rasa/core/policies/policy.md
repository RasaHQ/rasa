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
 | trackers_for_policy(policy: Union["Policy", Type["Policy"]], trackers: List[DialogueStateTracker]) -> List[DialogueStateTracker]
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
- `domain` - the :class:`rasa.core.domain.Domain`
- `interpreter` - the :class:`rasa.core.interpreter.NaturalLanguageInterpreter`
  

**Returns**:

  - a dictionary of attribute (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
  ENTITIES, SLOTS, FORM) to a list of features for all dialogue turns in
  all training trackers
  - the label ids (e.g. action ids) for every dialuge turn in all training
  trackers

#### train

```python
 | train(training_trackers: List[TrackerWithCachedStates], domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> None
```

Trains the policy on given training trackers.

**Arguments**:

  training_trackers:
  the list of the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.core.domain.Domain`
- `interpreter` - Interpreter which can be used by the polices for featurization.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> List[float]
```

Predicts the next action the bot should take after seeing the tracker.

**Arguments**:

- `tracker` - the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.core.domain.Domain`
- `interpreter` - Interpreter which may be used by the policies to create
  additional features.
  

**Returns**:

  the list of probabilities for the next actions

#### persist

```python
 | persist(path: Text) -> None
```

Persists the policy to a storage.

**Arguments**:

- `path` - the path where to save the policy to

#### load

```python
 | @classmethod
 | load(cls, path: Text) -> "Policy"
```

Loads a policy from the storage.

Needs to load its featurizer.

**Arguments**:

- `path` - the path from where to load the policy

#### confidence\_scores\_for

```python
confidence_scores_for(action_name: Text, value: float, domain: Domain) -> List[float]
```

Returns confidence scores if a single action is predicted.

**Arguments**:

- `action_name` - the name of the action for which the score should be set
- `value` - the confidence for `action_name`
- `domain` - the :class:`rasa.core.domain.Domain`
  

**Returns**:

  the list of the length of the number of actions

