---
sidebar_label: rasa.core.policies.form_policy
title: rasa.core.policies.form_policy
---
## FormPolicy Objects

```python
class FormPolicy(MemoizationPolicy)
```

Policy which handles prediction of Forms

#### recall

```python
 | recall(states: List[State], tracker: DialogueStateTracker, domain: Domain) -> Optional[Text]
```

Finds the action based on the given states.

**Arguments**:

- `states` - List of states.
- `tracker` - The tracker.
- `domain` - The Domain.
  

**Returns**:

  The name of the action.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the corresponding form action if there is an active form.

