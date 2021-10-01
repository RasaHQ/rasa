---
sidebar_label: rasa.core.policies._memoization
title: rasa.core.policies._memoization
---
## MemoizationPolicy Objects

```python
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

#### \_\_init\_\_

```python
def __init__(featurizer: Optional[TrackerFeaturizer] = None, priority: int = MEMOIZATION_POLICY_PRIORITY, max_history: Optional[int] = DEFAULT_MAX_HISTORY, lookup: Optional[Dict] = None, **kwargs: Any, ,) -> None
```

Initialize the policy.

**Arguments**:

- `featurizer` - tracker featurizer
- `priority` - the priority of the policy
- `max_history` - maximum history to take into account when featurizing trackers
- `lookup` - a dictionary that stores featurized tracker states and
  predicted actions for them

#### recall

```python
def recall(states: List[State], tracker: DialogueStateTracker, domain: Domain) -> Optional[Text]
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
def predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the next action the bot should take after seeing the tracker.

**Arguments**:

- `tracker` - the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `interpreter` - Interpreter which may be used by the policies to create
  additional features.
  

**Returns**:

  The policy&#x27;s prediction (e.g. the probabilities for the actions).

## AugmentedMemoizationPolicy Objects

```python
class AugmentedMemoizationPolicy(MemoizationPolicy)
```

The policy that remembers examples from training stories
for `max_history` turns.

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
def recall(states: List[State], tracker: DialogueStateTracker, domain: Domain) -> Optional[Text]
```

Finds the action based on the given states.

Uses back to the future idea to change the past and check whether the new future
can be used to recall the action.

**Arguments**:

- `states` - List of states.
- `tracker` - The tracker.
- `domain` - The Domain.
  

**Returns**:

  The name of the action.

