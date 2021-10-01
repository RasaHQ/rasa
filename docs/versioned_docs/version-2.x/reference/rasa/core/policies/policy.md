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

#### \_\_init\_\_

```python
 | __init__(featurizer: Optional[TrackerFeaturizer] = None, priority: int = DEFAULT_POLICY_PRIORITY, should_finetune: bool = False, **kwargs: Any, ,) -> None
```

Constructs a new Policy object.

#### featurizer

```python
 | @property
 | featurizer() -> TrackerFeaturizer
```

Returns the policy&#x27;s featurizer.

#### set\_shared\_policy\_states

```python
 | set_shared_policy_states(**kwargs: Any) -> None
```

Sets policy&#x27;s shared states for correct featurization.

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
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> "PolicyPrediction"
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
 | persist(path: Union[Text, Path]) -> None
```

Persists the policy to storage.

**Arguments**:

- `path` - Path to persist policy to.

#### load

```python
 | @classmethod
 | load(cls, path: Union[Text, Path], **kwargs: Any) -> "Policy"
```

Loads a policy from path.

**Arguments**:

- `path` - Path to load policy from.
  

**Returns**:

  An instance of `Policy`.

#### format\_tracker\_states

```python
 | @staticmethod
 | format_tracker_states(states: List[Dict]) -> Text
```

Format tracker states to human readable format on debug log.

**Arguments**:

- `states` - list of tracker states dicts
  

**Returns**:

  the string of the states with user intents and actions

## PolicyPrediction Objects

```python
class PolicyPrediction()
```

Stores information about the prediction of a `Policy`.

#### \_\_init\_\_

```python
 | __init__(probabilities: List[float], policy_name: Optional[Text], policy_priority: int = 1, events: Optional[List[Event]] = None, optional_events: Optional[List[Event]] = None, is_end_to_end_prediction: bool = False, is_no_user_prediction: bool = False, diagnostic_data: Optional[Dict[Text, Any]] = None, hide_rule_turn: bool = False, action_metadata: Optional[Dict[Text, Any]] = None) -> None
```

Creates a `PolicyPrediction`.

**Arguments**:

- `probabilities` - The probabilities for each action.
- `policy_name` - Name of the policy which made the prediction.
- `policy_priority` - The priority of the policy which made the prediction.
- `events` - Events which the `Policy` needs to have applied to the tracker
  after the prediction. These events are applied independent of whether
  the policy wins against other policies or not. Be careful which events
  you return as they can potentially influence the conversation flow.
- `optional_events` - Events which the `Policy` needs to have applied to the
  tracker after the prediction in case it wins. These events are only
  applied in case the policy&#x27;s prediction wins. Be careful which events
  you return as they can potentially influence the conversation flow.
- `is_end_to_end_prediction` - `True` if the prediction used the text of the
  user message instead of the intent.
- `is_no_user_prediction` - `True` if the prediction uses neither the text
  of the user message nor the intent. This is for the example the case
  for happy loop paths.
- `diagnostic_data` - Intermediate results or other information that is not
  necessary for Rasa to function, but intended for debugging and
  fine-tuning purposes.
- `hide_rule_turn` - `True` if the prediction was made by the rules which
  do not appear in the stories
- `action_metadata` - Specifies additional metadata that can be passed
  by policies.

#### for\_action\_name

```python
 | @staticmethod
 | for_action_name(domain: Domain, action_name: Text, policy_name: Optional[Text] = None, confidence: float = 1.0, action_metadata: Optional[Dict[Text, Any]] = None) -> "PolicyPrediction"
```

Create a prediction for a given action.

**Arguments**:

- `domain` - The current model domain
- `action_name` - The action which should be predicted.
- `policy_name` - The policy which did the prediction.
- `confidence` - The prediction confidence.
- `action_metadata` - Additional metadata to be attached with the prediction.
  

**Returns**:

  The prediction.

#### \_\_eq\_\_

```python
 | __eq__(other: Any) -> bool
```

Checks if the two objects are equal.

**Arguments**:

- `other` - Any other object.
  

**Returns**:

  `True` if other has the same type and the values are the same.

#### max\_confidence\_index

```python
 | @property
 | max_confidence_index() -> int
```

Gets the index of the action prediction with the highest confidence.

**Returns**:

  The index of the action with the highest confidence.

#### max\_confidence

```python
 | @property
 | max_confidence() -> float
```

Gets the highest predicted probability.

**Returns**:

  The highest predicted probability.

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

