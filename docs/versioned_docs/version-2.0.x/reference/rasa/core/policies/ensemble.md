---
sidebar_label: rasa.core.policies.ensemble
title: rasa.core.policies.ensemble
---

## PolicyEnsemble Objects

```python
class PolicyEnsemble()
```

#### check\_domain\_ensemble\_compatibility

```python
 | @staticmethod
 | check_domain_ensemble_compatibility(ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]) -> None
```

Check for elements that only work with certain policy/domain combinations.

#### persist

```python
 | persist(path: Union[Text, Path]) -> None
```

Persists the policy to storage.

#### load

```python
 | @classmethod
 | load(cls, path: Union[Text, Path]) -> "PolicyEnsemble"
```

Loads policy and domain specification from storage

## Prediction Objects

```python
class Prediction(NamedTuple)
```

Stores the probabilities and the priority of the prediction.

## SimplePolicyEnsemble Objects

```python
class SimplePolicyEnsemble(PolicyEnsemble)
```

#### probabilities\_using\_best\_policy

```python
 | probabilities_using_best_policy(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> Tuple[List[float], Optional[Text]]
```

Predicts the next action the bot should take after seeing the tracker.

Picks the best policy prediction based on probabilities and policy priority.
Triggers fallback if `action_listen` is predicted after a user utterance.

**Arguments**:

- `tracker` - the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `interpreter` - Interpreter which may be used by the policies to create
  additional features.
  

**Returns**:

- `best_probabilities` - the list of probabilities for the next actions
- `best_policy_name` - the name of the picked policy

## InvalidPolicyConfig Objects

```python
class InvalidPolicyConfig(RasaException)
```

Exception that can be raised when policy config is not valid.

