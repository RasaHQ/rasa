---
sidebar_label: rasa.utils.tensorflow.data_generator
title: rasa.utils.tensorflow.data_generator
---
## RasaDataGenerator Objects

```python
class RasaDataGenerator(tf.keras.utils.Sequence)
```

Abstract data generator.

#### \_\_init\_\_

```python
 | __init__(model_data: RasaModelData, batch_size: Union[int, List[int]], batch_strategy: Text = SEQUENCE, shuffle: bool = True)
```

Initializes the data generator.

**Arguments**:

- `model_data` - The model data to use.
- `batch_size` - The batch size(s).
- `batch_strategy` - The batch strategy.
- `shuffle` - If &#x27;True&#x27;, data should be shuffled.

#### \_\_len\_\_

```python
 | __len__() -> int
```

Number of batches in the Sequence.

**Returns**:

  The number of batches in the Sequence.

#### \_\_getitem\_\_

```python
 | __getitem__(index: int) -> Tuple[Any, Any]
```

Gets batch at position `index`.

**Arguments**:

- `index` - position of the batch in the Sequence.
  

**Returns**:

  A batch (tuple of input data and target data).

#### on\_epoch\_end

```python
 | on_epoch_end() -> None
```

Update the data after every epoch.

#### prepare\_batch

```python
 | @staticmethod
 | prepare_batch(data: Data, start: Optional[int] = None, end: Optional[int] = None, tuple_sizes: Optional[Dict[Text, int]] = None) -> Tuple[Optional[np.ndarray], ...]
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

## RasaBatchDataGenerator Objects

```python
class RasaBatchDataGenerator(RasaDataGenerator)
```

Data generator with an optional increasing batch size.

#### \_\_init\_\_

```python
 | __init__(model_data: RasaModelData, batch_size: Union[List[int], int], epochs: int = 1, batch_strategy: Text = SEQUENCE, shuffle: bool = True)
```

Initializes the increasing batch size data generator.

**Arguments**:

- `model_data` - The model data to use.
- `batch_size` - The batch size.
- `epochs` - The total number of epochs.
- `batch_strategy` - The batch strategy.
- `shuffle` - If &#x27;True&#x27;, data will be shuffled.

#### \_\_len\_\_

```python
 | __len__() -> int
```

Number of batches in the Sequence.

**Returns**:

  The number of batches in the Sequence.

#### \_\_getitem\_\_

```python
 | __getitem__(index: int) -> Tuple[Any, Any]
```

Gets batch at position `index`.

**Arguments**:

- `index` - position of the batch in the Sequence.
  

**Returns**:

  A batch (tuple of input data and target data).

#### on\_epoch\_end

```python
 | on_epoch_end() -> None
```

Update the data after every epoch.

