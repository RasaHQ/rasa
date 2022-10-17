---
sidebar_label: rasa.core.policies._rule_policy
title: rasa.core.policies._rule_policy
---
## InvalidRule Objects

```python
class InvalidRule(RasaException)
```

Exception that can be raised when rules are not valid.

## RulePolicy Objects

```python
class RulePolicy(MemoizationPolicy)
```

Policy which handles all the rules

#### supported\_data

```python
 | @staticmethod
 | supported_data() -> SupportedData
```

The type of data supported by this policy.

**Returns**:

  The data type supported by this policy (ML and rule data).

#### \_\_init\_\_

```python
 | __init__(featurizer: Optional[TrackerFeaturizer] = None, priority: int = RULE_POLICY_PRIORITY, lookup: Optional[Dict] = None, core_fallback_threshold: float = DEFAULT_CORE_FALLBACK_THRESHOLD, core_fallback_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME, enable_fallback_prediction: bool = True, restrict_rules: bool = True, check_for_contradictions: bool = True, **kwargs: Any, ,) -> None
```

Create a `RulePolicy` object.

**Arguments**:

- `featurizer` - `Featurizer` which is used to convert conversation states to
  features.
- `priority` - Priority of the policy which is used if multiple policies predict
  actions with the same confidence.
- `lookup` - Lookup table which is used to pick matching rules for a conversation
  state.
- `core_fallback_threshold` - Confidence of the prediction if no rule matched
  and de-facto threshold for a core fallback.
- `core_fallback_action_name` - Name of the action which should be predicted
  if no rule matched.
- `enable_fallback_prediction` - If `True` `core_fallback_action_name` is
  predicted in case no rule matched.
- `restrict_rules` - If `True` rules are restricted to contain a maximum of 1
  user message. This is used to avoid that users build a state machine
  using the rules.
- `check_for_contradictions` - Check for contradictions.

#### train

```python
 | train(training_trackers: List[TrackerWithCachedStates], domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> None
```

Trains the policy on given training trackers.

**Arguments**:

- `training_trackers` - The list of the trackers.
- `domain` - The domain.
- `interpreter` - Interpreter which can be used by the polices for featurization.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> "PolicyPrediction"
```

Predicts the next action (see parent class for more information).

#### get\_rule\_only\_data

```python
 | get_rule_only_data() -> Dict[Text, Any]
```

Gets the slots and loops that are used only in rule data.

**Returns**:

  Slots and loops that are used only in rule data.

