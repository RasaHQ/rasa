---
sidebar_label: ted_policy
title: rasa.core.policies.ted_policy
---

## TEDPolicy Objects

```python
class TEDPolicy(Policy)
```

Transformer Embedding Dialogue (TED) Policy is described in
https://arxiv.org/abs/1910.00486.
This policy has a pre-defined architecture, which comprises the
following steps:
    - concatenate user input (user intent and entities), previous system actions,
      slots and active forms for each time step into an input vector to
      pre-transformer embedding layer;
    - feed it to transformer;
    - apply a dense layer to the output of the transformer to get embeddings of a
      dialogue for each time step;
    - apply a dense layer to create embeddings for system actions for each time
      step;
    - calculate the similarity between the dialogue embedding and embedded system
      actions. This step is based on the StarSpace
      (https://arxiv.org/abs/1709.03856) idea.

#### \_\_init\_\_

```python
 | __init__(featurizer: Optional[TrackerFeaturizer] = None, priority: int = DEFAULT_POLICY_PRIORITY, max_history: Optional[int] = None, model: Optional[RasaModel] = None, zero_state_features: Optional[Dict[Text, List["Features"]]] = None, should_finetune: bool = False, **kwargs: Any, ,) -> None
```

Declare instance variables with default values.

#### train

```python
 | train(training_trackers: List[TrackerWithCachedStates], domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> None
```

Train the policy on given training trackers.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the next action the bot should take.

See the docstring of the parent class `Policy` for more information.

#### persist

```python
 | persist(path: Union[Text, Path]) -> None
```

Persists the policy to a storage.

#### load

```python
 | @classmethod
 | load(cls, path: Union[Text, Path], should_finetune: bool = False, epoch_override: int = defaults[EPOCHS], **kwargs: Any, ,) -> "TEDPolicy"
```

Loads a policy from the storage.

**Needs to load its featurizer**

