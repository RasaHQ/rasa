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
def __init__(features: Union[np.ndarray, scipy.sparse.spmatrix], feature_type: Text, attribute: Text, origin: Union[Text, List[Text]]) -> None
```

Initializes the Features object.

**Arguments**:

- `features` - The features.
- `feature_type` - Type of the feature, e.g. FEATURE_TYPE_SENTENCE.
- `attribute` - Message attribute, e.g. INTENT or TEXT.
- `origin` - Name of the component that created the features.

#### is\_sparse

```python
def is_sparse() -> bool
```

Checks if features are sparse or not.

**Returns**:

  True, if features are sparse, false otherwise.

#### is\_dense

```python
def is_dense() -> bool
```

Checks if features are dense or not.

**Returns**:

  True, if features are dense, false otherwise.

#### combine\_with\_features

```python
def combine_with_features(additional_features: Optional[Features]) -> None
```

Combine the incoming features with this instance&#x27;s features.

**Arguments**:

- `additional_features` - additional features to add
  

**Returns**:

  Combined features.

#### \_\_key\_\_

```python
def __key__() -> Tuple[
        Text, Text, Union[np.ndarray, scipy.sparse.spmatrix], Union[Text, List[Text]]
    ]
```

Returns a 4-tuple of defining properties.

**Returns**:

  Tuple of type, attribute, features, and origin properties.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Tests if the `self` `Feature` equals to the `other`.

**Arguments**:

- `other` - The other object.
  

**Returns**:

  `True` when the other object is a `Feature` and has the same
  type, attribute, and feature tensors.

#### fingerprint

```python
def fingerprint() -> Text
```

Calculate a stable string fingerprint for the features.

#### filter

```python
@staticmethod
def filter(features_list: List[Features], attributes: Optional[Iterable[Text]] = None, type: Optional[Text] = None, origin: Optional[List[Text]] = None, is_sparse: Optional[bool] = None) -> List[Features]
```

Filters the given list of features.

**Arguments**:

- `features_list` - list of features to be filtered
- `attributes` - List of attributes that we&#x27;re interested in. Set this to `None`
  to disable this filter.
- `type` - The type of feature we&#x27;re interested in. Set this to `None`
  to disable this filter.
- `origin` - If specified, this method will check that the exact order of origins
  matches the given list of origins. The reason for this is that if
  multiple origins are listed for a Feature, this means that this feature
  has been created by concatenating Features from the listed origins in
  that particular order.
- `is_sparse` - Defines whether all features that we&#x27;re interested in should be
  sparse. Set this to `None` to disable this filter.
  

**Returns**:

  sub-list of features with the desired properties

#### groupby\_attribute

```python
@staticmethod
def groupby_attribute(features_list: List[Features], attributes: Optional[Iterable[Text]] = None) -> Dict[Text, List[Features]]
```

Groups the given features according to their attribute.

**Arguments**:

- `features_list` - list of features to be grouped
- `attributes` - If specified, the result will be a grouping with respect to
  the given attributes. If some specified attribute has no features attached
  to it, then the resulting dictionary will map it to an empty list.
  If this is None, the result will be a grouping according to all attributes
  for which features can be found.
  

**Returns**:

  a mapping from the requested attributes to the list of correspoding
  features

#### combine

```python
@staticmethod
def combine(features_list: List[Features], expected_origins: Optional[List[Text]] = None) -> Features
```

Combine features of the same type and level that describe the same attribute.

If sequence features are to be combined, then they must have the same
sequence dimension.

**Arguments**:

- `features` - Non-empty list of Features  of the same type and level that
  describe the same attribute.
- `expected_origins` - The expected origins of the given features. This method
  will check that the origin information of each feature is as expected, i.e.
  the origin of the i-th feature in the given list is the i-th origin
  in this list of origins.
  

**Raises**:

  `ValueError` will be raised
  - if the given list is empty
  - if there are inconsistencies in the given list of `Features`
  - if the origins aren&#x27;t as expected

#### reduce

```python
@staticmethod
def reduce(features_list: List[Features], expected_origins: Optional[List[Text]] = None) -> List[Features]
```

Combines features of same type and level into one Feature.

**Arguments**:

- `features_list` - list of Features which must all describe the same attribute
- `expected_origins` - if specified, this list will be used to validate that
  the features from the right featurizers are combined in the right order
  (cf. `Features.combine`)
  

**Returns**:

  a list of the combined Features, i.e. at most 4 Features, where
  - all the sparse features are listed before the dense features
  - sequence feature is always listed before the sentence feature with the
  same sparseness property

