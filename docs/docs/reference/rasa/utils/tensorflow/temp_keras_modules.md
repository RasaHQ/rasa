---
sidebar_label: rasa.utils.tensorflow.temp_keras_modules
title: rasa.utils.tensorflow.temp_keras_modules
---
## TmpKerasModel Objects

```python
class TmpKerasModel(Model)
```

Temporary solution. Keras model that uses a custom data adapter inside fit.

#### fit

```python
 | @traceback_utils.filter_traceback
 | fit(x: Optional[
 |             Union[np.ndarray, tf.Tensor, tf.data.Dataset, tf.keras.utils.Sequence]
 |         ] = None, y: Optional[
 |             Union[np.ndarray, tf.Tensor, tf.data.Dataset, tf.keras.utils.Sequence]
 |         ] = None, batch_size: Optional[int] = None, epochs: int = 1, verbose: int = 1, callbacks: Optional[List[Callback]] = None, validation_split: float = 0.0, validation_data: Optional[Any] = None, shuffle: bool = True, class_weight: Optional[Dict[int, float]] = None, sample_weight: Optional[np.ndarray] = None, initial_epoch: int = 0, steps_per_epoch: Optional[int] = None, validation_steps: Optional[int] = None, validation_batch_size: Optional[int] = None, validation_freq: int = 1, max_queue_size: int = 10, workers: int = 1, use_multiprocessing: bool = False) -> History
```

Trains the model for a fixed number of epochs (iterations on a dataset).

**Arguments**:

- `x` - Input data. It could be:
  - A Numpy array (or array-like), or a list of arrays
  (in case the model has multiple inputs).
  - A TensorFlow tensor, or a list of tensors
  (in case the model has multiple inputs).
  - A dict mapping input names to the corresponding array/tensors,
  if the model has named inputs.
  - A `tf.data` dataset. Should return a tuple
  of either `(inputs, targets)` or
  `(inputs, targets, sample_weights)`.
  - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
  or `(inputs, targets, sample_weights)`.
  - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
  callable that takes a single argument of type
  `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
  `DatasetCreator` should be used when users prefer to specify the
  per-replica batching and sharding logic for the `Dataset`.
  See `tf.keras.utils.experimental.DatasetCreator` doc for more
  information.
  A more detailed description of unpacking behavior for iterator types
  (Dataset, generator, Sequence) is given below. If using
  `tf.distribute.experimental.ParameterServerStrategy`, only
  `DatasetCreator` type is supported for `x`.
- `y` - Target data. Like the input data `x`,
  it could be either Numpy array(s) or TensorFlow tensor(s).
  It should be consistent with `x` (you cannot have Numpy inputs and
  tensor targets, or inversely). If `x` is a dataset, generator,
  or `keras.utils.Sequence` instance, `y` should
  not be specified (since targets will be obtained from `x`).
- `batch_size` - Integer or `None`.
  Number of samples per gradient update.
  If unspecified, `batch_size` will default to 32.
  Do not specify the `batch_size` if your data is in the
  form of datasets, generators, or `keras.utils.Sequence` instances
  (since they generate batches).
- `epochs` - Integer. Number of epochs to train the model.
  An epoch is an iteration over the entire `x` and `y`
  data provided
  (unless the `steps_per_epoch` flag is set to
  something other than None).
  Note that in conjunction with `initial_epoch`,
  `epochs` is to be understood as &quot;final epoch&quot;.
  The model is not trained for a number of iterations
  given by `epochs`, but merely until the epoch
  of index `epochs` is reached.
- `verbose` - &#x27;auto&#x27;, 0, 1, or 2. Verbosity mode.
  0 = silent, 1 = progress bar, 2 = one line per epoch.
  &#x27;auto&#x27; defaults to 1 for most cases, but 2 when used with
  `ParameterServerStrategy`. Note that the progress bar is not
  particularly useful when logged to a file, so verbose=2 is
  recommended when not running interactively (eg, in a production
  environment).
- `callbacks` - List of `keras.callbacks.Callback` instances.
  List of callbacks to apply during training.
  See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
  and `tf.keras.callbacks.History` callbacks are created automatically
  and need not be passed into `model.fit`.
  `tf.keras.callbacks.ProgbarLogger` is created or not based on
  `verbose` argument to `model.fit`.
  Callbacks with batch-level calls are currently unsupported with
  `tf.distribute.experimental.ParameterServerStrategy`, and users are
  advised to implement epoch-level calls instead with an appropriate
  `steps_per_epoch` value.
- `validation_split` - Float between 0 and 1.
  Fraction of the training data to be used as validation data.
  The model will set apart this fraction of the training data,
  will not train on it, and will evaluate
  the loss and any model metrics
  on this data at the end of each epoch.
  The validation data is selected from the last samples
  in the `x` and `y` data provided, before shuffling. This argument is
  not supported when `x` is a dataset, generator or
  `keras.utils.Sequence` instance.
  `validation_split` is not yet supported with
  `tf.distribute.experimental.ParameterServerStrategy`.
- `validation_data` - Data on which to evaluate
  the loss and any model metrics at the end of each epoch.
  The model will not be trained on this data. Thus, note the fact
  that the validation loss of data provided using `validation_split`
  or `validation_data` is not affected by regularization layers like
  noise and dropout.
  `validation_data` will override `validation_split`.
  `validation_data` could be:
  - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
  - A tuple `(x_val, y_val, val_sample_weights)` of NumPy arrays.
  - A `tf.data.Dataset`.
  - A Python generator or `keras.utils.Sequence` returning
  `(inputs, targets)` or `(inputs, targets, sample_weights)`.
  `validation_data` is not yet supported with
  `tf.distribute.experimental.ParameterServerStrategy`.
- `shuffle` - Boolean (whether to shuffle the training data
  before each epoch) or str (for &#x27;batch&#x27;). This argument is ignored
  when `x` is a generator or an object of tf.data.Dataset.
  &#x27;batch&#x27; is a special option for dealing
  with the limitations of HDF5 data; it shuffles in batch-sized
  chunks. Has no effect when `steps_per_epoch` is not `None`.
- `class_weight` - Optional dictionary mapping class indices (integers)
  to a weight (float) value, used for weighting the loss function
  (during training only).
  This can be useful to tell the model to
  &quot;pay more attention&quot; to samples from
  an under-represented class.
- `sample_weight` - Optional Numpy array of weights for
  the training samples, used for weighting the loss function
  (during training only). You can either pass a flat (1D)
  Numpy array with the same length as the input samples
  (1:1 mapping between weights and samples),
  or in the case of temporal data,
  you can pass a 2D array with shape
  `(samples, sequence_length)`,
  to apply a different weight to every timestep of every sample. This
  argument is not supported when `x` is a dataset, generator, or
  `keras.utils.Sequence` instance, instead provide the sample_weights
  as the third element of `x`.
- `initial_epoch` - Integer.
  Epoch at which to start training
  (useful for resuming a previous training run).
- `steps_per_epoch` - Integer or `None`.
  Total number of steps (batches of samples)
  before declaring one epoch finished and starting the
  next epoch. When training with input tensors such as
  TensorFlow data tensors, the default `None` is equal to
  the number of samples in your dataset divided by
  the batch size, or 1 if that cannot be determined. If x is a
  `tf.data` dataset, and &#x27;steps_per_epoch&#x27;
  is None, the epoch will run until the input dataset is exhausted.
  When passing an infinitely repeating dataset, you must specify the
  `steps_per_epoch` argument. If `steps_per_epoch=-1` the training
  will run indefinitely with an infinitely repeating dataset.
  This argument is not supported with array inputs.
  When using `tf.distribute.experimental.ParameterServerStrategy`:
  * `steps_per_epoch=None` is not supported.
- `validation_steps` - Only relevant if `validation_data` is provided and
  is a `tf.data` dataset. Total number of steps (batches of
  samples) to draw before stopping when performing validation
  at the end of every epoch. If &#x27;validation_steps&#x27; is None, validation
  will run until the `validation_data` dataset is exhausted. In the
  case of an infinitely repeated dataset, it will run into an
  infinite loop. If &#x27;validation_steps&#x27; is specified and only part of
  the dataset will be consumed, the evaluation will start from the
  beginning of the dataset at each epoch. This ensures that the same
  validation samples are used every time.
- `validation_batch_size` - Integer or `None`.
  Number of samples per validation batch.
  If unspecified, will default to `batch_size`.
  Do not specify the `validation_batch_size` if your data is in the
  form of datasets, generators, or `keras.utils.Sequence` instances
  (since they generate batches).
- `validation_freq` - Only relevant if validation data is provided. Integer
  or `collections.abc.Container` instance (e.g. list, tuple, etc.).
  If an integer, specifies how many training epochs to run before a
  new validation run is performed, e.g. `validation_freq=2` runs
  validation every 2 epochs. If a Container, specifies the epochs on
  which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
  validation at the end of the 1st, 2nd, and 10th epochs.
- `max_queue_size` - Integer. Used for generator or `keras.utils.Sequence`
  input only. Maximum size for the generator queue.
  If unspecified, `max_queue_size` will default to 10.
- `workers` - Integer. Used for generator or `keras.utils.Sequence` input
  only. Maximum number of processes to spin up
  when using process-based threading. If unspecified, `workers`
  will default to 1.
- `use_multiprocessing` - Boolean. Used for generator or
  `keras.utils.Sequence` input only. If `True`, use process-based
  threading. If unspecified, `use_multiprocessing` will default to
  `False`. Note that because this implementation relies on
  multiprocessing, you should not pass non-picklable arguments to
  the generator as they can&#x27;t be passed easily to children processes.
  Unpacking behavior for iterator-like inputs:
  A common pattern is to pass a tf.data.Dataset, generator, or
  tf.keras.utils.Sequence to the `x` argument of fit, which will in fact
  yield not only features (x) but optionally targets (y) and sample weights.
  Keras requires that the output of such iterator-likes be unambiguous. The
  iterator should return a tuple of length 1, 2, or 3, where the optional
  second and third elements will be used for y and sample_weight
  respectively. Any other type provided will be wrapped in a length one
  tuple, effectively treating everything as &#x27;x&#x27;. When yielding dicts, they
  should still adhere to the top-level tuple structure.
  e.g. `({&quot;x0&quot;: x0, &quot;x1&quot;: x1}, y)`. Keras will not attempt to separate
  features, targets, and weights from the keys of a single dict.
  A notable unsupported data type is the namedtuple. The reason is that
  it behaves like both an ordered datatype (tuple) and a mapping
  datatype (dict). So given a namedtuple of the form:
  `namedtuple(&quot;example_tuple&quot;, [&quot;y&quot;, &quot;x&quot;])`
  it is ambiguous whether to reverse the order of the elements when
  interpreting the value. Even worse is a tuple of the form:
  `namedtuple(&quot;other_tuple&quot;, [&quot;x&quot;, &quot;y&quot;, &quot;z&quot;])`
  where it is unclear if the tuple was intended to be unpacked into x, y,
  and sample_weight or passed through as a single element to `x`. As a
  result the data processing code will simply raise a ValueError if it
  encounters a namedtuple. (Along with instructions to remedy the issue.)
  

**Returns**:

  A `History` object. Its `History.history` attribute is
  a record of training loss values and metrics values
  at successive epochs, as well as validation loss values
  and validation metrics values (if applicable).
  

**Raises**:

- `RuntimeError` - 1. If the model was never compiled or,
  2. If `model.fit` is  wrapped in `tf.function`.
- `ValueError` - In case of mismatch between the provided input data
  and what the model expects or when the input data is empty.

## CustomDataHandler Objects

```python
class CustomDataHandler(data_adapter.DataHandler)
```

Handles iterating over epoch-level `tf.data.Iterator` objects.

#### enumerate\_epochs

```python
 | enumerate_epochs() -> Generator[Tuple[int, Iterator], None, None]
```

Yields `(epoch, tf.data.Iterator)`.

