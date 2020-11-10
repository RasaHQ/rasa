---
sidebar_label: rasa.core.policies.form_policy
title: rasa.core.policies.form_policy
---

## FormPolicy Objects

```python
class FormPolicy(MemoizationPolicy)
```

Policy which handles prediction of Forms

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the corresponding form action if there is an active form.

