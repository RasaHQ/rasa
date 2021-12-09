---
sidebar_label: rasa.core.policies.sklearn_policy
title: rasa.core.policies.sklearn_policy
---
## SklearnPolicy Objects

```python
class SklearnPolicy(Policy)
```

Use an sklearn classifier to train a policy.

#### \_\_init\_\_

```python
 | __init__(featurizer: Optional[MaxHistoryTrackerFeaturizer] = None, priority: int = DEFAULT_POLICY_PRIORITY, max_history: int = DEFAULT_MAX_HISTORY, model: Optional["sklearn.base.BaseEstimator"] = None, param_grid: Optional[Union[Dict[Text, List], List[Dict]]] = None, cv: Optional[int] = None, scoring: Optional[Union[Text, List, Dict, Callable]] = "accuracy", label_encoder: LabelEncoder = LabelEncoder(), shuffle: bool = True, zero_state_features: Optional[Dict[Text, List["Features"]]] = None, **kwargs: Any, ,) -> None
```

Create a new sklearn policy.

**Arguments**:

- `featurizer` - Featurizer used to convert the training data into
  vector format.
- `priority` - Policy priority
- `max_history` - Maximum history of the dialogs.
- `model` - The sklearn model or model pipeline.
- `param_grid` - If *param_grid* is not None and *cv* is given,
  a grid search on the given *param_grid* is performed
  (e.g. *param_grid={&#x27;n_estimators&#x27;: [50, 100]}*).
- `cv` - If *cv* is not None, perform a cross validation on
  the training data. *cv* should then conform to the
  sklearn standard (e.g. *cv=5* for a 5-fold cross-validation).
- `scoring` - Scoring strategy, using the sklearn standard.
- `label_encoder` - Encoder for the labels. Must implement an
  *inverse_transform* method.
- `shuffle` - Whether to shuffle training data.
- `zero_state_features` - Contains default feature values for attributes

#### model\_architecture

```python
 | model_architecture(**kwargs: Any) -> Any
```

Sets model parameters for training.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the next action the bot should take after seeing the tracker.

**Arguments**:

- `tracker` - the :class:`rasa.core.trackers.DialogueStateTracker`
- `domain` - the :class:`rasa.shared.core.domain.Domain`
- `interpreter` - Interpreter which may be used by the policies to create
  additional features.
  

**Returns**:

  The policy&#x27;s prediction (e.g. the probabilities for the actions).

#### persist

```python
 | persist(path: Union[Text, Path]) -> None
```

Persists the policy properties (see parent class for more information).

#### load

```python
 | @classmethod
 | load(cls, path: Union[Text, Path], should_finetune: bool = False, **kwargs: Any) -> Policy
```

See the docstring for `Policy.load`.

