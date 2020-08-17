---
sidebar_label: rasa.core.policies.rule_policy
title: rasa.core.policies.rule_policy
---

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

  The data type supported by this policy (rule data).

#### \_\_init\_\_

```python
 | __init__(featurizer: Optional[TrackerFeaturizer] = None, priority: int = FORM_POLICY_PRIORITY, lookup: Optional[Dict] = None, core_fallback_threshold: float = 0.3, core_fallback_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME, enable_fallback_prediction: bool = True) -> None
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

