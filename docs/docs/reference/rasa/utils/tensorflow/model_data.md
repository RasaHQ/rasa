---
sidebar_label: rasa.utils.tensorflow.model_data
title: rasa.utils.tensorflow.model_data
---

## FeatureSignature Objects

```python
class FeatureSignature(NamedTuple)
```

Stores the shape and the type (sparse vs dense) of features.

## RasaModelData Objects

```python
class RasaModelData()
```

Data object used for all RasaModels.

It contains all features needed to train the models.

#### \_\_init\_\_

```python
 | __init__(label_key: Optional[Text] = None, data: Optional[Data] = None) -> None
```

Initializes the RasaModelData object.

**Arguments**:

- `label_key` - the label_key used for balancing, etc.
- `data` - the data holding the features

#### feature\_not\_exist

```python
 | feature_not_exist(key: Text) -> bool
```

Check if feature key is present and features are available.

#### is\_empty

```python
 | is_empty() -> bool
```

Checks if data is set.

#### number\_of\_examples

```python
 | number_of_examples(data: Optional[Data] = None) -> int
```

Obtain number of examples in data.

Raises: A ValueError if number of examples differ for different features.

#### feature\_dimension

```python
 | feature_dimension(key: Text) -> int
```

Get the feature dimension of the given key.

#### add\_features

```python
 | add_features(key: Text, features: List[np.ndarray])
```

Add list of features to data under specified key.

Should update number of examples.

#### add\_lengths

```python
 | add_lengths(key: Text, from_key: Text) -> None
```

Adds np.array of lengths of sequences to data under given key.

#### split

```python
 | split(number_of_test_examples: int, random_seed: int) -> Tuple["RasaModelData", "RasaModelData"]
```

Create random hold out test set using stratified split.

#### get\_signature

```python
 | get_signature() -> Dict[Text, List[FeatureSignature]]
```

Get signature of RasaModelData.

Signature stores the shape and whether features are sparse or not for every key.

#### as\_tf\_dataset

```python
 | as_tf_dataset(batch_size: int, batch_strategy: Text = SEQUENCE, shuffle: bool = False) -> tf.data.Dataset
```

Create tf dataset.

#### prepare\_batch

```python
 | prepare_batch(data: Optional[Data] = None, start: Optional[int] = None, end: Optional[int] = None, tuple_sizes: Optional[Dict[Text, int]] = None) -> Tuple[Optional[np.ndarray]]
```

Slices model data into batch using given start and end value.

