---
sidebar_label: rasa.utils.features
title: rasa.utils.features
---
## Features Objects

```python
class Features()
```

Stores the features produces by any featurizer.

#### is\_sparse

```python
 | is_sparse() -> bool
```

Checks if features are sparse or not.

**Returns**:

  True, if features are sparse, false otherwise.

#### is\_dense

```python
 | is_dense() -> bool
```

Checks if features are dense or not.

**Returns**:

  True, if features are dense, false otherwise.

#### combine\_with\_features

```python
 | combine_with_features(additional_features: Optional["Features"]) -> None
```

Combine the incoming features with this instance&#x27;s features.

**Arguments**:

- `additional_features` - additional features to add
  

**Returns**:

  Combined features.

