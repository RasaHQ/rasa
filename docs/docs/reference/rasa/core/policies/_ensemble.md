---
sidebar_label: rasa.core.policies._ensemble
title: rasa.core.policies._ensemble
---
## PolicyEnsemble Objects

```python
class PolicyEnsemble()
```

#### check\_domain\_ensemble\_compatibility

```python
@staticmethod
def check_domain_ensemble_compatibility(ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]) -> None
```

Check for elements that only work with certain policy/domain combinations.

#### persist

```python
def persist(path: Union[Text, Path]) -> None
```

Persists the policy to storage.

#### load

```python
@classmethod
def load(cls, path: Union[Text, Path], new_config: Optional[Dict] = None, finetuning_epoch_fraction: float = 1.0) -> "PolicyEnsemble"
```

Loads policy and domain specification from disk.

#### get\_featurizer\_from\_dict

```python
@classmethod
def get_featurizer_from_dict(cls, policy: Dict[Text, Any]) -> Tuple[Any, Any]
```

Gets the featurizer initializer and its arguments from a policy config.

## SimplePolicyEnsemble Objects

```python
class SimplePolicyEnsemble(PolicyEnsemble)
```

Default implementation of a `Policy` ensemble.

#### is\_not\_in\_training\_data

```python
@staticmethod
def is_not_in_training_data(policy_name: Optional[Text], max_confidence: Optional[float] = None) -> bool
```

Checks if the prediction is by a policy which memoized the training data.

**Arguments**:

- `policy_name` - The name of the policy.
- `max_confidence` - The max confidence of the policy&#x27;s prediction.
  
- `Returns` - `True` if it&#x27;s a `MemoizationPolicy`, `False` otherwise.

#### probabilities\_using\_best\_policy

```python
def probabilities_using_best_policy(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
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

  The best policy prediction.

## InvalidPolicyConfig Objects

```python
class InvalidPolicyConfig(RasaException)
```

Exception that can be raised when policy config is not valid.

