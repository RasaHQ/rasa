---
sidebar_label: model_data
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
 | __init__(label_key: Optional[Text] = None, label_sub_key: Optional[Text] = None, data: Optional[Data] = None) -> None
```

Initializes the RasaModelData object.

**Arguments**:

- `label_key` - the key of a label used for balancing, etc.
- `label_sub_key` - the sub key of a label used for balancing, etc.
- `data` - the data holding the features

#### get

```python
 | get(key: Text, sub_key: Optional[Text] = None) -> Union[Dict[Text, List[np.ndarray]], List[np.ndarray]]
```

Get the data under the given keys.

**Arguments**:

- `key` - The key.
- `sub_key` - The optional sub key.
  

**Returns**:

  The requested data.

#### items

```python
 | items() -> ItemsView
```

Return the items of the data attribute.

**Returns**:

  The items of data.

#### values

```python
 | values() -> ValuesView[Dict[Text, List[np.ndarray]]]
```

Return the values of the data attribute.

**Returns**:

  The values of data.

#### keys

```python
 | keys(key: Optional[Text] = None) -> List[Text]
```

Return the keys of the data attribute.

**Arguments**:

- `key` - The optional key.
  

**Returns**:

  The keys of the data.

#### first\_data\_example

```python
 | first_data_example() -> Data
```

Return the data with just one feature example per key, sub-key.

**Returns**:

  The simplified data.

#### does\_feature\_not\_exist

```python
 | does_feature_not_exist(key: Text, sub_key: Optional[Text] = None) -> bool
```

Check if feature key (and sub-key) is present and features are available.

**Arguments**:

- `key` - The key.
- `sub_key` - The optional sub-key.
  

**Returns**:

  True, if no features for the given keys exists, False otherwise.

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

**Arguments**:

- `data` - The data.
  
- `Raises` - A ValueError if number of examples differ for different features.
  

**Returns**:

  The number of examples in data.

#### feature\_dimension

```python
 | feature_dimension(key: Text, sub_key: Text) -> int
```

Get the feature dimension of the given key.

**Arguments**:

- `key` - The key.
- `sub_key` - The optional sub-key.
  

**Returns**:

  The feature dimension.

#### add\_data

```python
 | add_data(data: Data, key_prefix: Optional[Text] = None) -> None
```

Add incoming data to data.

**Arguments**:

- `data` - The data to add.
- `key_prefix` - Optional key prefix to use in front of the key value.

#### add\_features

```python
 | add_features(key: Text, sub_key: Text, features: Optional[List[np.ndarray]]) -> None
```

Add list of features to data under specified key.

Should update number of examples.

**Arguments**:

- `key` - The key
- `sub_key` - The sub-key
- `features` - The features to add.

#### add\_lengths

```python
 | add_lengths(key: Text, sub_key: Text, from_key: Text, from_sub_key: Text) -> None
```

Adds np.array of lengths of sequences to data under given key.

**Arguments**:

- `key` - The key to add the lengths to
- `sub_key` - The sub-key to add the lengths to
- `from_key` - The key to take the lengths from
- `from_sub_key` - The sub-key to take the lengths from

#### split

```python
 | split(number_of_test_examples: int, random_seed: int) -> Tuple["RasaModelData", "RasaModelData"]
```

Create random hold out test set using stratified split.

**Arguments**:

- `number_of_test_examples` - Number of test examples.
- `random_seed` - Random seed.
  

**Returns**:

  A tuple of train and test RasaModelData.

#### get\_signature

```python
 | get_signature() -> Dict[Text, Dict[Text, List[FeatureSignature]]]
```

Get signature of RasaModelData.

Signature stores the shape and whether features are sparse or not for every key.

**Returns**:

  A dictionary of key and sub-key to a list of feature signatures
  (same structure as the data attribute).

#### as\_tf\_dataset

```python
 | as_tf_dataset(batch_size: int, batch_strategy: Text = SEQUENCE, shuffle: bool = False) -> tf.data.Dataset
```

Create tf dataset.

**Arguments**:

- `batch_size` - The batch size to use.
- `batch_strategy` - The batch strategy to use.
- `shuffle` - Boolean indicating whether the data should be shuffled or not.
  

**Returns**:

  The tf.data.Dataset.

#### prepare\_batch

```python
 | prepare_batch(data: Optional[Data] = None, start: Optional[int] = None, end: Optional[int] = None, tuple_sizes: Optional[Dict[Text, int]] = None) -> Tuple[Optional[np.ndarray]]
```

Slices model data into batch using given start and end value.

**Arguments**:

- `data` - The data to prepare.
- `start` - The start index of the batch
- `end` - The end index of the batch
- `tuple_sizes` - In case the feature is not present we propagate the batch with
  None. Tuple sizes contains the number of how many None values to add for
  what kind of feature.
  

**Returns**:

  The features of the batch.

