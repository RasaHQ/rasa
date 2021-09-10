---
sidebar_label: rasa.utils.tensorflow.model_data
title: rasa.utils.tensorflow.model_data
---
## FeatureArray Objects

```python
class FeatureArray(np.ndarray)
```

Stores any kind of features ready to be used by a RasaModel.

Next to the input numpy array of features, it also received the number of
dimensions of the features.
As our features can have 1 to 4 dimensions we might have different number of numpy
arrays stacked. The number of dimensions helps us to figure out how to handle this
particular feature array. Also, it is automatically determined whether the feature
array is sparse or not and the number of units is determined as well.

Subclassing np.array: https://numpy.org/doc/stable/user/basics.subclassing.html

#### \_\_new\_\_

```python
def __new__(cls, input_array: np.ndarray, number_of_dimensions: int) -> "FeatureArray"
```

Create and return a new object.  See help(type) for accurate signature.

#### \_\_init\_\_

```python
def __init__(input_array: Any, number_of_dimensions: int, **kwargs: Any) -> None
```

Initialize. FeatureArray.

Needed in order to avoid &#x27;Invalid keyword argument number_of_dimensions
to function FeatureArray.__init__ &#x27;

**Arguments**:

- `input_array` - the array that contains features
- `number_of_dimensions` - number of dimensions in input_array

#### \_\_array\_finalize\_\_

```python
def __array_finalize__(obj: Any) -> None
```

This method is called when the system allocates a new array from obj.

**Arguments**:

- `obj` - A subclass (subtype) of ndarray.

#### \_\_array\_ufunc\_\_

```python
def __array_ufunc__(ufunc: Any, method: Text, *inputs: Any, **kwargs: Any) -> Any
```

Overwrite this method as we are subclassing numpy array.

**Arguments**:

- `ufunc` - The ufunc object that was called.
- `method` - A string indicating which Ufunc method was called
  (one of &quot;__call__&quot;, &quot;reduce&quot;, &quot;reduceat&quot;, &quot;accumulate&quot;, &quot;outer&quot;,
  &quot;inner&quot;).
- `*inputs` - A tuple of the input arguments to the ufunc.
- `**kwargs` - Any additional arguments
  

**Returns**:

  The result of the operation.

#### \_\_reduce\_\_

```python
def __reduce__() -> Tuple[Any, Any, Any]
```

Needed in order to pickle this object.

**Returns**:

  A tuple.

#### \_\_setstate\_\_

```python
def __setstate__(state: Any, **kwargs: Any) -> None
```

Sets the state.

**Arguments**:

- `state` - The state argument must be a sequence that contains the following
  elements version, shape, dtype, isFortan, rawdata.
- `**kwargs` - Any additional parameter

## FeatureSignature Objects

```python
class FeatureSignature(NamedTuple)
```

Signature of feature arrays.

Stores the number of units, the type (sparse vs dense), and the number of
dimensions of features.

## RasaModelData Objects

```python
class RasaModelData()
```

Data object used for all RasaModels.

It contains all features needed to train the models.
&#x27;data&#x27; is a mapping of attribute name, e.g. TEXT, INTENT, etc., and feature name,
e.g. SENTENCE, SEQUENCE, etc., to a list of feature arrays representing the actual
features.
&#x27;label_key&#x27; and &#x27;label_sub_key&#x27; point to the labels inside &#x27;data&#x27;. For
example, if your intent labels are stored under INTENT -&gt; IDS, &#x27;label_key&#x27; would
be &quot;INTENT&quot; and &#x27;label_sub_key&#x27; would be &quot;IDS&quot;.

#### \_\_init\_\_

```python
def __init__(label_key: Optional[Text] = None, label_sub_key: Optional[Text] = None, data: Optional[Data] = None) -> None
```

Initializes the RasaModelData object.

**Arguments**:

- `label_key` - the key of a label used for balancing, etc.
- `label_sub_key` - the sub key of a label used for balancing, etc.
- `data` - the data holding the features

#### get

```python
def get(key: Text, sub_key: Optional[Text] = None) -> Union[Dict[Text, List[FeatureArray]], List[FeatureArray]]
```

Get the data under the given keys.

**Arguments**:

- `key` - The key.
- `sub_key` - The optional sub key.
  

**Returns**:

  The requested data.

#### items

```python
def items() -> ItemsView
```

Return the items of the data attribute.

**Returns**:

  The items of data.

#### values

```python
def values() -> Any
```

Return the values of the data attribute.

**Returns**:

  The values of data.

#### keys

```python
def keys(key: Optional[Text] = None) -> List[Text]
```

Return the keys of the data attribute.

**Arguments**:

- `key` - The optional key.
  

**Returns**:

  The keys of the data.

#### sort

```python
def sort() -> None
```

Sorts data according to its keys.

#### first\_data\_example

```python
def first_data_example() -> Data
```

Return the data with just one feature example per key, sub-key.

**Returns**:

  The simplified data.

#### does\_feature\_exist

```python
def does_feature_exist(key: Text, sub_key: Optional[Text] = None) -> bool
```

Check if feature key (and sub-key) is present and features are available.

**Arguments**:

- `key` - The key.
- `sub_key` - The optional sub-key.
  

**Returns**:

  False, if no features for the given keys exists, True otherwise.

#### does\_feature\_not\_exist

```python
def does_feature_not_exist(key: Text, sub_key: Optional[Text] = None) -> bool
```

Check if feature key (and sub-key) is present and features are available.

**Arguments**:

- `key` - The key.
- `sub_key` - The optional sub-key.
  

**Returns**:

  True, if no features for the given keys exists, False otherwise.

#### is\_empty

```python
def is_empty() -> bool
```

Checks if data is set.

#### number\_of\_examples

```python
def number_of_examples(data: Optional[Data] = None) -> int
```

Obtain number of examples in data.

**Arguments**:

- `data` - The data.
  
- `Raises` - A ValueError if number of examples differ for different features.
  

**Returns**:

  The number of examples in data.

#### number\_of\_units

```python
def number_of_units(key: Text, sub_key: Text) -> int
```

Get the number of units of the given key.

**Arguments**:

- `key` - The key.
- `sub_key` - The optional sub-key.
  

**Returns**:

  The number of units.

#### add\_data

```python
def add_data(data: Data, key_prefix: Optional[Text] = None) -> None
```

Add incoming data to data.

**Arguments**:

- `data` - The data to add.
- `key_prefix` - Optional key prefix to use in front of the key value.

#### update\_key

```python
def update_key(from_key: Text, from_sub_key: Text, to_key: Text, to_sub_key: Text) -> None
```

Copies the features under the given keys to the new keys and deletes the old.

**Arguments**:

- `from_key` - current feature key
- `from_sub_key` - current feature sub-key
- `to_key` - new key for feature
- `to_sub_key` - new sub-key for feature

#### add\_features

```python
def add_features(key: Text, sub_key: Text, features: Optional[List[FeatureArray]]) -> None
```

Add list of features to data under specified key.

Should update number of examples.

**Arguments**:

- `key` - The key
- `sub_key` - The sub-key
- `features` - The features to add.

#### add\_lengths

```python
def add_lengths(key: Text, sub_key: Text, from_key: Text, from_sub_key: Text) -> None
```

Adds a feature array of lengths of sequences to data under given key.

**Arguments**:

- `key` - The key to add the lengths to
- `sub_key` - The sub-key to add the lengths to
- `from_key` - The key to take the lengths from
- `from_sub_key` - The sub-key to take the lengths from

#### add\_sparse\_feature\_sizes

```python
def add_sparse_feature_sizes(sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]]) -> None
```

Adds a dictionary of feature sizes for different attributes.

**Arguments**:

- `sparse_feature_sizes` - a dictionary of attribute that has sparse
  features to a dictionary of a feature type
  to a list of different sparse feature sizes.

#### get\_sparse\_feature\_sizes

```python
def get_sparse_feature_sizes() -> Dict[Text, Dict[Text, List[int]]]
```

Get feature sizes of the model.

sparse_feature_sizes is a dictionary of attribute that has sparse features to
a dictionary of a feature type to a list of different sparse feature sizes.

**Returns**:

  A dictionary of key and sub-key to a list of feature signatures
  (same structure as the data attribute).

#### split

```python
def split(number_of_test_examples: int, random_seed: int) -> Tuple["RasaModelData", "RasaModelData"]
```

Create random hold out test set using stratified split.

**Arguments**:

- `number_of_test_examples` - Number of test examples.
- `random_seed` - Random seed.
  

**Returns**:

  A tuple of train and test RasaModelData.

#### get\_signature

```python
def get_signature(data: Optional[Data] = None) -> Dict[Text, Dict[Text, List[FeatureSignature]]]
```

Get signature of RasaModelData.

Signature stores the shape and whether features are sparse or not for every key.

**Returns**:

  A dictionary of key and sub-key to a list of feature signatures
  (same structure as the data attribute).

#### shuffled\_data

```python
def shuffled_data(data: Data) -> Data
```

Shuffle model data.

**Arguments**:

- `data` - The data to shuffle
  

**Returns**:

  The shuffled data.

#### balanced\_data

```python
def balanced_data(data: Data, batch_size: int, shuffle: bool) -> Data
```

Mix model data to account for class imbalance.

This batching strategy puts rare classes approximately in every other batch,
by repeating them. Mimics stratified batching, but also takes into account
that more populated classes should appear more often.

**Arguments**:

- `data` - The data.
- `batch_size` - The batch size.
- `shuffle` - Boolean indicating whether to shuffle the data or not.
  

**Returns**:

  The balanced data.

