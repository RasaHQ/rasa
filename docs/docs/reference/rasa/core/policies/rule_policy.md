---
sidebar_label: rasa.core.policies.rule_policy
title: rasa.core.policies.rule_policy
---
## InvalidRule Objects

```python
class InvalidRule(RasaException)
```

Exception that can be raised when rules are not valid.

## RulePolicyGraphComponent Objects

```python
class RulePolicyGraphComponent(MemoizationPolicyGraphComponent)
```

Policy which handles all the rules.

#### supported\_data

```python
 | @staticmethod
 | supported_data() -> SupportedData
```

The type of data supported by this policy.

**Returns**:

  The data type supported by this policy (ML and rule data).

#### get\_default\_config

```python
 | @staticmethod
 | get_default_config() -> Dict[Text, Any]
```

Returns the default config (see parent class for full docstring).

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, featurizer: Optional[TrackerFeaturizer] = None, lookup: Optional[Dict] = None) -> None
```

Initializes the policy.

#### train

```python
 | train(training_trackers: List[TrackerWithCachedStates], domain: Domain, **kwargs: Any, ,) -> Resource
```

Trains the policy on given training trackers.

**Arguments**:

- `training_trackers` - The list of the trackers.
- `domain` - The domain.
  

**Returns**:

  The resource which can be used to load the trained policy.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, rule_only_data: Optional[Dict[Text, Any]] = None, **kwargs: Any, ,) -> "PolicyPrediction"
```

Predicts the next action (see parent class for more information).

#### persist

```python
 | persist() -> None
```

Persists trained `RulePolicy`.

