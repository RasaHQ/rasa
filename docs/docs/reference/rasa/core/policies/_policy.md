---
sidebar_label: rasa.core.policies._policy
title: rasa.core.policies._policy
---
## Policy Objects

```python
class Policy()
```

Common parent class for all dialogue policies.

#### supported\_data

```python
@staticmethod
def supported_data() -> "SupportedData"
```

The type of data supported by this policy.

By default, this is only ML-based training data. If policies support rule data,
or both ML-based data and rule data, they need to override this method.

**Returns**:

  The data type supported by this policy (ML-based training data).

#### \_\_init\_\_

```python
def __init__(featurizer: Optional[TrackerFeaturizer] = None, priority: int = DEFAULT_POLICY_PRIORITY, should_finetune: bool = False, **kwargs: Any, ,) -> None
```

Constructs a new Policy object.

#### featurizer

```python
@property
def featurizer() -> TrackerFeaturizer
```

Returns the policy&#x27;s featurizer.

#### set\_shared\_policy\_states

```python
def set_shared_policy_states(**kwargs: Any) -> None
```

Sets policy&#x27;s shared states for correct featurization.

#### train

```python
def train(training_trackers: List[TrackerWithCachedStates], domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> None
```

Trains the policy on given training trackers.

**Arguments**:

  training_trackers:
  the list of the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `interpreter` - Interpreter which can be used by the polices for featurization.
- `**kwargs` - Additional keyword arguments.

#### predict\_action\_probabilities

```python
def predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> "PolicyPrediction"
```

Predicts the next action the bot should take after seeing the tracker.

**Arguments**:

- `tracker` - the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `interpreter` - Interpreter which may be used by the policies to create
  additional features.
  

**Returns**:

  The policy&#x27;s prediction (e.g. the probabilities for the actions).

#### persist

```python
def persist(path: Union[Text, Path]) -> None
```

Persists the policy to storage.

**Arguments**:

- `path` - Path to persist policy to.

#### load

```python
@classmethod
def load(cls, path: Union[Text, Path], **kwargs: Any) -> "Policy"
```

Loads a policy from path.

**Arguments**:

- `path` - Path to load policy from.
  

**Returns**:

  An instance of `Policy`.

#### format\_tracker\_states

```python
@staticmethod
def format_tracker_states(states: List[Dict]) -> Text
```

Format tracker states to human readable format on debug log.

**Arguments**:

- `states` - list of tracker states dicts
  

**Returns**:

  the string of the states with user intents and actions

