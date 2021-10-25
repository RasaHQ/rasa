---
sidebar_label: rasa.core.policies.memoization
title: rasa.core.policies.memoization
---
## MemoizationPolicy Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=True
)
class MemoizationPolicy(Policy)
```

A policy that follows exact examples of `max_history` turns in training stories.

Since `slots` that are set some time in the past are
preserved in all future feature vectors until they are set
to None, this policy implicitly remembers and most importantly
recalls examples in the context of the current dialogue
longer than `max_history`.

This policy is not supposed to be the only policy in an ensemble,
it is optimized for precision and not recall.
It should get a 100% precision because it emits probabilities of 1.1
along it&#x27;s predictions, which makes every mistake fatal as
no other policy can overrule it.

If it is needed to recall turns from training dialogues where
some slots might not be set during prediction time, and there are
training stories for this, use AugmentedMemoizationPolicy.

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

Initialize the policy.

#### recall

```python
 | recall(states: List[State], tracker: DialogueStateTracker, domain: Domain, rule_only_data: Optional[Dict[Text, Any]]) -> Optional[Text]
```

Finds the action based on the given states.

**Arguments**:

- `states` - List of states.
- `tracker` - The tracker.
- `domain` - The Domain.
- `rule_only_data` - Slots and loops which are specific to rules and hence
  should be ignored by this policy.
  

**Returns**:

  The name of the action.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, rule_only_data: Optional[Dict[Text, Any]] = None, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the next action the bot should take after seeing the tracker.

**Arguments**:

- `tracker` - the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `rule_only_data` - Slots and loops which are specific to rules and hence
  should be ignored by this policy.
  

**Returns**:

  The policy&#x27;s prediction (e.g. the probabilities for the actions).

#### persist

```python
 | persist() -> None
```

Persists the policy to storage.

#### load

```python
 | @classmethod
 | load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> MemoizationPolicy
```

Loads a trained policy (see parent class for full docstring).

## AugmentedMemoizationPolicy Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=True
)
class AugmentedMemoizationPolicy(MemoizationPolicy)
```

The policy that remembers examples from training stories for `max_history` turns.

If it is needed to recall turns from training dialogues
where some slots might not be set during prediction time,
add relevant stories without such slots to training data.
E.g. reminder stories.

Since `slots` that are set some time in the past are
preserved in all future feature vectors until they are set
to None, this policy has a capability to recall the turns
up to `max_history` from training stories during prediction
even if additional slots were filled in the past
for current dialogue.

#### recall

```python
 | recall(states: List[State], tracker: DialogueStateTracker, domain: Domain, rule_only_data: Optional[Dict[Text, Any]]) -> Optional[Text]
```

Finds the action based on the given states.

Uses back to the future idea to change the past and check whether the new future
can be used to recall the action.

**Arguments**:

- `states` - List of states.
- `tracker` - The tracker.
- `domain` - The Domain.
- `rule_only_data` - Slots and loops which are specific to rules and hence
  should be ignored by this policy.
  

**Returns**:

  The name of the action.

