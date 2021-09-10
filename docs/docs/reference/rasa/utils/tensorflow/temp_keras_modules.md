---
sidebar_label: rasa.utils.tensorflow.temp_keras_modules
title: rasa.utils.tensorflow.temp_keras_modules
---
## TmpKerasModel Objects

```python
class TmpKerasModel(tf.keras.models.Model)
```

Temporary solution. Keras model that uses a custom data adapter inside fit.

#### fit

```python
@training.enable_multi_worker
def fit(x: Optional[
            Union[np.ndarray, tf.Tensor, tf.data.Dataset, tf.keras.utils.Sequence]
        ] = None, y: Optional[
            Union[np.ndarray, tf.Tensor, tf.data.Dataset, tf.keras.utils.Sequence]
        ] = None, batch_size: Optional[int] = None, epochs: int = 1, verbose: int = 1, callbacks: Optional[List[Callback]] = None, validation_split: float = 0.0, validation_data: Optional[Any] = None, shuffle: bool = True, class_weight: Optional[Dict[int, float]] = None, sample_weight: Optional[np.ndarray] = None, initial_epoch: int = 0, steps_per_epoch: Optional[int] = None, validation_steps: Optional[int] = None, validation_batch_size: Optional[int] = None, validation_freq: int = 1, max_queue_size: int = 10, workers: int = 1, use_multiprocessing: bool = False) -> History
```

Trains the model for a fixed number of epochs (iterations on a dataset).

**Arguments**:

- `x` - Input data.
- `y` - Target data.
- `batch_size` - Number of samples per gradient update.
- `epochs` - Number of epochs to train the model.
- `verbose` - Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per
  epoch.
- `callbacks` - List of `keras.callbacks.Callback` instances.
- `validation_split` - Fraction of the training data to be used as validation
  data.
- `validation_data` - Data on which to evaluate the loss and any model metrics
  at the end of each epoch.
- `shuffle` - whether to shuffle the training data before each epoch
- `class_weight` - Optional dictionary mapping class indices (integers)
  to a weight (float) value, used for weighting the loss
  function (during training only).
- `sample_weight` - Optional Numpy array of weights for
  the training samples, used for weighting the loss function
  (during training only).
- `initial_epoch` - Epoch at which to start training
- `steps_per_epoch` - Total number of steps (batches of samples)
  before declaring one epoch finished and starting the
  next epoch.
- `validation_steps` - Total number of steps (batches of
  samples) to draw before stopping when performing
  validation at the end of every epoch.
- `validation_batch_size` - Number of samples per validation batch.
- `validation_freq` - specifies how many training epochs to run before a
  new validation run is performed
- `max_queue_size` - Maximum size for the generator queue.
- `workers` - Maximum number of processes to spin up
  when using process-based threading.
- `use_multiprocessing` - If `True`, use process-based threading.
  

**Returns**:

  A `History` object. Its `History.history` attribute is
  a record of training loss values and metrics values
  at successive epochs, as well as validation loss values
  and validation metrics values (if applicable).
  

**Raises**:

- `RuntimeError` - 1. If the model was never compiled or,
  2. If `model.fit` is  wrapped in `tf.function`.
  
- `ValueError` - In case of mismatch between the provided input data
  and what the model expects.

## CustomDataHandler Objects

```python
class CustomDataHandler(DataHandler)
```

Handles iterating over epoch-level `tf.data.Iterator` objects.

#### enumerate\_epochs

```python
def enumerate_epochs() -> Generator[Tuple[int, Iterator], None, None]
```

Yields `(epoch, tf.data.Iterator)`.

