---
sidebar_label: rasa.shared.nlu.training_data.features
title: rasa.shared.nlu.training_data.features
---
## Features Objects

```python
class Features()
```

Stores the features produced by any featurizer.

#### \_\_init\_\_

```python
 | __init__(features: Union[np.ndarray, scipy.sparse.spmatrix], feature_type: Text, attribute: Text, origin: Union[Text, List[Text]]) -> None
```

Initializes the Features object.

**Arguments**:

- `features` - The features.
- `feature_type` - Type of the feature, e.g. FEATURE_TYPE_SENTENCE.
- `attribute` - Message attribute, e.g. INTENT or TEXT.
- `origin` - Name of the component that created the features.

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

#### \_\_key\_\_

```python
 | __key__() -> Tuple[
 |         Text, Text, Union[np.ndarray, scipy.sparse.spmatrix], Union[Text, List[Text]]
 |     ]
```

Returns a 4-tuple of defining properties.

**Returns**:

  Tuple of type, attribute, features, and origin properties.

#### \_\_eq\_\_

```python
 | __eq__(other: Any) -> bool
```

Tests if the `self` `Feature` equals to the `other`.

**Arguments**:

- `other` - The other object.
  

**Returns**:

  `True` when the other object is a `Feature` and has the same
  type, attribute, and feature tensors.

