---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.dense_featurizer
title: rasa.nlu.featurizers.dense_featurizer.dense_featurizer
---
## DenseFeaturizer2 Objects

```python
class DenseFeaturizer2(Featurizer2[np.ndarray],  ABC)
```

Base class for all dense featurizers.

#### aggregate\_sequence\_features

```python
@staticmethod
def aggregate_sequence_features(dense_sequence_features: np.ndarray, pooling_operation: Text, only_non_zero_vectors: bool = True) -> np.ndarray
```

Aggregates the non-zero vectors of a dense sequence feature matrix.

**Arguments**:

- `dense_sequence_features` - a 2-dimensional matrix where the first dimension
  is the sequence dimension over which we want to aggregate of shape
  [seq_len, feat_dim]
- `pooling_operation` - either max pooling or average pooling
- `only_non_zero_vectors` - determines whether the aggregation is done over
  non-zero vectors only

**Returns**:

  a matrix of shape [1, feat_dim]

