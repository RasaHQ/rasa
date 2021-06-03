---
sidebar_label: rasa.core.policies.mapping_policy
title: rasa.core.policies.mapping_policy
---
## MappingPolicy Objects

```python
class MappingPolicy(Policy)
```

Policy which maps intents directly to actions.

Intents can be assigned actions in the domain file which are to be
executed whenever the intent is detected. This policy takes precedence over
any other policy.

#### \_\_init\_\_

```python
 | __init__(priority: int = MAPPING_POLICY_PRIORITY, **kwargs: Any) -> None
```

Create a new Mapping policy.

#### train

```python
 | train(training_trackers: List[TrackerWithCachedStates], domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> None
```

Does nothing. This policy is deterministic.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the assigned action.

If the current intent is assigned to an action that action will be
predicted with the highest probability of all policies. If it is not
the policy will predict zero for every action.

