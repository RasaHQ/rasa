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
@staticmethod
def trackers_for_policy(policy: Union[Policy, Type[Policy]], trackers: Union[List[DialogueStateTracker], List[TrackerWithCachedStates]]) -> Union[List[DialogueStateTracker], List[TrackerWithCachedStates]]
```

Return trackers for a given policy.

**Arguments**:

- `policy` - Policy or policy type to return trackers for.
- `trackers` - Trackers to split.
  

**Returns**:

  Trackers from ML-based training data and/or rule-based data.

## PolicyGraphComponent Objects

```python
class PolicyGraphComponent(GraphComponent)
```

Common parent class for all dialogue policies.

#### supported\_data

```python
@staticmethod
def supported_data() -> SupportedData
```

The type of data supported by this policy.

By default, this is only ML-based training data. If policies support rule data,
or both ML-based data and rule data, they need to override this method.

**Returns**:

  The data type supported by this policy (ML-based training data).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, featurizer: Optional[TrackerFeaturizer] = None) -> None
```

Constructs a new Policy object.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> PolicyGraphComponent
```

Creates a new untrained policy (see parent class for full docstring).

#### featurizer

```python
@property
def featurizer() -> TrackerFeaturizer
```

Returns the policy&#x27;s featurizer.

#### train

```python
@abc.abstractmethod
def train(training_trackers: List[TrackerWithCachedStates], domain: Domain, **kwargs: Any, ,) -> Resource
```

Trains a policy.

**Arguments**:

- `training_trackers` - The story and rules trackers from the training data.
- `domain` - The model&#x27;s domain.
- `**kwargs` - Depending on the specified `needs` section and the resulting
  graph structure the policy can use different input to train itself.
  

**Returns**:

  A policy must return its resource locator so that potential children nodes
  can load the policy from the resource.

#### predict\_action\_probabilities

```python
@abc.abstractmethod
def predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, rule_only_data: Optional[Dict[Text, Any]] = None, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the next action the bot should take after seeing the tracker.

**Arguments**:

- `tracker` - The tracker containing the conversation history up to now.
- `domain` - The model&#x27;s domain.
- `rule_only_data` - Slots and loops which are specific to rules and hence
  should be ignored by this policy.
- `**kwargs` - Depending on the specified `needs` section and the resulting
  graph structure the policy can use different input to make predictions.
  

**Returns**:

  The prediction.

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> "PolicyGraphComponent"
```

Loads a trained policy (see parent class for full docstring).

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

## PolicyPrediction Objects

```python
class PolicyPrediction()
```

Stores information about the prediction of a `Policy`.

#### \_\_init\_\_

```python
def __init__(probabilities: List[float], policy_name: Optional[Text], policy_priority: int = 1, events: Optional[List[Event]] = None, optional_events: Optional[List[Event]] = None, is_end_to_end_prediction: bool = False, is_no_user_prediction: bool = False, diagnostic_data: Optional[Dict[Text, Any]] = None, hide_rule_turn: bool = False, action_metadata: Optional[Dict[Text, Any]] = None) -> None
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
@staticmethod
def for_action_name(domain: Domain, action_name: Text, policy_name: Optional[Text] = None, confidence: float = 1.0, action_metadata: Optional[Dict[Text, Any]] = None) -> "PolicyPrediction"
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
def __eq__(other: Any) -> bool
```

Checks if the two objects are equal.

**Arguments**:

- `other` - Any other object.
  

**Returns**:

  `True` if other has the same type and the values are the same.

#### max\_confidence\_index

```python
@property
def max_confidence_index() -> int
```

Gets the index of the action prediction with the highest confidence.

**Returns**:

  The index of the action with the highest confidence.

#### max\_confidence

```python
@property
def max_confidence() -> float
```

Gets the highest predicted confidence.

**Returns**:

  The highest predicted confidence.

#### confidence\_scores\_for

```python
def confidence_scores_for(action_name: Text, value: float, domain: Domain) -> List[float]
```

Returns confidence scores if a single action is predicted.

**Arguments**:

- `action_name` - the name of the action for which the score should be set
- `value` - the confidence for `action_name`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
  

**Returns**:

  the list of the length of the number of actions

## InvalidPolicyConfig Objects

```python
class InvalidPolicyConfig(RasaException)
```

Exception that can be raised when policy config is not valid.

