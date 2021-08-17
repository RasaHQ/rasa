---
sidebar_label: sklearn_policy
title: rasa.core.policies.sklearn_policy
---

## SklearnPolicy Objects

```python
class SklearnPolicy(Policy)
```

Use an sklearn classifier to train a policy.

#### \_\_init\_\_

```python
 | __init__(featurizer: Optional[MaxHistoryTrackerFeaturizer] = None, priority: int = DEFAULT_POLICY_PRIORITY, max_history: int = DEFAULT_MAX_HISTORY, model: Optional["sklearn.base.BaseEstimator"] = None, param_grid: Optional[Dict[Text, List] or List[Dict]] = None, cv: Optional[int] = None, scoring: Optional[Text or List or Dict or Callable] = "accuracy", label_encoder: LabelEncoder = LabelEncoder(), shuffle: bool = True, zero_state_features: Optional[Dict[Text, List["Features"]]] = None, **kwargs: Any, ,) -> None
```

Create a new sklearn policy.

**Arguments**:

- `featurizer` - Featurizer used to convert the training data into
  vector format.
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

#### persist

```python
 | persist(path: Union[Text, Path]) -> None
```

Persists the policy properties (see parent class for more information).

