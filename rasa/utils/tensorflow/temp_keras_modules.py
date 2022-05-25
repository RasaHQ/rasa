import copy
from typing import List, Dict, Union, Optional, Any, Generator, Tuple, Iterator, cast

import numpy as np

import tensorflow as tf

# Note: the below is the same as `from keras.engine.training import Model`
from tensorflow.keras import Model

# Note: the following import is from the temporary/non-public tf.python module but
# should stay since it's used in the current reference implementation (see link below)
from tensorflow.python.eager import context

# Note: the following imports are  from keras directly since those are the imports
# used by the reference implementation:
# https://github.com/keras-team/keras/blob/v2.7.0/keras/engine/training.py#L30
from keras.engine import base_layer, training_utils, data_adapter
from keras import callbacks as callbacks_module
from keras.callbacks import Callback, History
from keras.utils import traceback_utils, tf_utils, version_utils


def _disallow_inside_tf_function(method_name: str) -> None:
    if tf.inside_function():
        error_msg = (
            "Detected a call to `Model.{method_name}` inside a `tf.function`. "
            "`Model.{method_name} is a high-level endpoint that manages its own "
            "`tf.function`. Please move the call to `Model.{method_name}` outside "
            "of all enclosing `tf.function`s. Note that you can call a `Model` "
            "directly on `Tensor`s inside a `tf.function` like: `model(x)`."
        ).format(method_name=method_name)
        raise RuntimeError(error_msg)


# noinspection PyMethodOverriding
class TmpKerasModel(Model):
    """Temporary solution. Keras model that uses a custom data adapter inside fit."""

    # TODO
    #  we don't need this anymore once the fix from
    #  https://github.com/tensorflow/tensorflow/pull/45338
    #  has been ported over to keras and merged there

    # This code is adapted from
    # https://github.com/keras-team/keras/blob/v2.7.0/keras/engine/training.py#L902

    @traceback_utils.filter_traceback  # type: ignore[misc]
    def fit(
        self,
        x: Optional[
            Union[np.ndarray, tf.Tensor, tf.data.Dataset, tf.keras.utils.Sequence]
        ] = None,
        y: Optional[
            Union[np.ndarray, tf.Tensor, tf.data.Dataset, tf.keras.utils.Sequence]
        ] = None,
        batch_size: Optional[int] = None,
        epochs: int = 1,
        verbose: int = 1,
        callbacks: Optional[List[Callback]] = None,
        validation_split: float = 0.0,
        validation_data: Optional[Any] = None,
        shuffle: bool = True,
        class_weight: Optional[Dict[int, float]] = None,
        sample_weight: Optional[np.ndarray] = None,
        initial_epoch: int = 0,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        validation_batch_size: Optional[int] = None,
        validation_freq: int = 1,
        max_queue_size: int = 10,
        workers: int = 1,
        use_multiprocessing: bool = False,
    ) -> History:
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            x: Input data. It could be:
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
            y: Target data. Like the input data `x`,
              it could be either Numpy array(s) or TensorFlow tensor(s).
              It should be consistent with `x` (you cannot have Numpy inputs and
              tensor targets, or inversely). If `x` is a dataset, generator,
              or `keras.utils.Sequence` instance, `y` should
              not be specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of datasets, generators, or `keras.utils.Sequence` instances
                (since they generate batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided
                (unless the `steps_per_epoch` flag is set to
                something other than None).
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 'auto', 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                'auto' defaults to 1 for most cases, but 2 when used with
                `ParameterServerStrategy`. Note that the progress bar is not
                particularly useful when logged to a file, so verbose=2 is
                recommended when not running interactively (eg, in a production
                environment).
            callbacks: List of `keras.callbacks.Callback` instances.
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
            validation_split: Float between 0 and 1.
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
            validation_data: Data on which to evaluate
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
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch'). This argument is ignored
                when `x` is a generator or an object of tf.data.Dataset.
                'batch' is a special option for dealing
                with the limitations of HDF5 data; it shuffles in batch-sized
                chunks. Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
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
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined. If x is a
                `tf.data` dataset, and 'steps_per_epoch'
                is None, the epoch will run until the input dataset is exhausted.
                When passing an infinitely repeating dataset, you must specify the
                `steps_per_epoch` argument. If `steps_per_epoch=-1` the training
                will run indefinitely with an infinitely repeating dataset.
                This argument is not supported with array inputs.
                When using `tf.distribute.experimental.ParameterServerStrategy`:
                  * `steps_per_epoch=None` is not supported.
            validation_steps: Only relevant if `validation_data` is provided and
                is a `tf.data` dataset. Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If 'validation_steps' is None, validation
                will run until the `validation_data` dataset is exhausted. In the
                case of an infinitely repeated dataset, it will run into an
                infinite loop. If 'validation_steps' is specified and only part of
                the dataset will be consumed, the evaluation will start from the
                beginning of the dataset at each epoch. This ensures that the same
                validation samples are used every time.
            validation_batch_size: Integer or `None`.
                Number of samples per validation batch.
                If unspecified, will default to `batch_size`.
                Do not specify the `validation_batch_size` if your data is in the
                form of datasets, generators, or `keras.utils.Sequence` instances
                (since they generate batches).
            validation_freq: Only relevant if validation data is provided. Integer
                or `collections.abc.Container` instance (e.g. list, tuple, etc.).
                If an integer, specifies how many training epochs to run before a
                new validation run is performed, e.g. `validation_freq=2` runs
                validation every 2 epochs. If a Container, specifies the epochs on
                which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
                validation at the end of the 1st, 2nd, and 10th epochs.
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up
                when using process-based threading. If unspecified, `workers`
                will default to 1.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
        Unpacking behavior for iterator-like inputs:
            A common pattern is to pass a tf.data.Dataset, generator, or
          tf.keras.utils.Sequence to the `x` argument of fit, which will in fact
          yield not only features (x) but optionally targets (y) and sample weights.
          Keras requires that the output of such iterator-likes be unambiguous. The
          iterator should return a tuple of length 1, 2, or 3, where the optional
          second and third elements will be used for y and sample_weight
          respectively. Any other type provided will be wrapped in a length one
          tuple, effectively treating everything as 'x'. When yielding dicts, they
          should still adhere to the top-level tuple structure.
          e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
          features, targets, and weights from the keys of a single dict.
            A notable unsupported data type is the namedtuple. The reason is that
          it behaves like both an ordered datatype (tuple) and a mapping
          datatype (dict). So given a namedtuple of the form:
              `namedtuple("example_tuple", ["y", "x"])`
          it is ambiguous whether to reverse the order of the elements when
          interpreting the value. Even worse is a tuple of the form:
              `namedtuple("other_tuple", ["x", "y", "z"])`
          where it is unclear if the tuple was intended to be unpacked into x, y,
          and sample_weight or passed through as a single element to `x`. As a
          result the data processing code will simply raise a ValueError if it
          encounters a namedtuple. (Along with instructions to remedy the issue.)

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        Raises:
            RuntimeError: 1. If the model was never compiled or,
            2. If `model.fit` is  wrapped in `tf.function`.
            ValueError: In case of mismatch between the provided input data
                and what the model expects or when the input data is empty.
        """
        base_layer.keras_api_gauge.get_cell("fit").set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph("Model", "fit")
        self._assert_compile_was_called()
        self._check_call_args("fit")
        _disallow_inside_tf_function("fit")

        if verbose == "auto":
            if (
                self.distribute_strategy._should_use_with_coordinator
            ):  # pylint: disable=protected-access
                verbose = 2  # Default to epoch-level logging for PSStrategy.
            else:
                verbose = 1  # Default to batch-level logging otherwise.
        elif (
            verbose == 1 and self.distribute_strategy._should_use_with_coordinator
        ):  # pylint: disable=protected-access
            raise ValueError(
                "`verbose=1` is not allowed with `ParameterServerStrategy` for "
                f"performance reasons. Received: `verbose`={verbose}"
            )

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Tensor` and `NumPy` input.
            (
                (x, y, sample_weight),
                validation_data,
            ) = data_adapter.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data:
            val_x, val_y, val_sample_weight = data_adapter.unpack_x_y_sample_weight(
                validation_data
            )

        if (
            self.distribute_strategy._should_use_with_coordinator
        ):  # pylint: disable=protected-access
            self._cluster_coordinator = (
                tf.distribute.experimental.coordinator.ClusterCoordinator(
                    self.distribute_strategy
                )
            )

        with self.distribute_strategy.scope(), training_utils.RespectCompiledTrainableState(  # noqa: E501
            self
        ):
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            # Adaption: Use our own custom data handler to handle increasing batch size
            data_handler = CustomDataHandler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution,
            )

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps,
                )
            callbacks_list = cast(callbacks_module.CallbackList, callbacks)

            self.stop_training = False
            self.train_function = self.make_train_function()
            self._train_counter.assign(0)
            callbacks_list.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            # TODO(omalleyt): Fix the ordering issues that mean this has to
            # happen after `callbacks.on_train_begin`.
            data_handler._initial_epoch = self._maybe_load_initial_epoch_from_ckpt(  # pylint: disable=protected-access # noqa: E501
                initial_epoch
            )
            logs = None
            for epoch, iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                callbacks_list.on_epoch_begin(epoch)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        with tf.profiler.experimental.Trace(
                            "train",
                            epoch_num=epoch,
                            step_num=step,
                            batch_size=batch_size,
                            _r=1,
                        ):
                            callbacks_list.on_train_batch_begin(step)
                            tmp_logs = self.train_function(iterator)
                            if data_handler.should_sync:
                                context.async_wait()
                            logs = tmp_logs  # No error, now safe to assign to logs.
                            end_step = step + data_handler.step_increment
                            callbacks_list.on_train_batch_end(end_step, logs)
                            if self.stop_training:
                                break

                logs = tf_utils.sync_to_numpy_or_python_type(logs)
                if logs is None:
                    raise ValueError(
                        "Unexpected result of `train_function` "
                        "(Empty logs). Please use "
                        "`Model.compile(..., run_eagerly=True)`, or "
                        "`tf.config.run_functions_eagerly(True)` for more "
                        "information of where went wrong, or file a "
                        "issue/bug to `tf.keras`."
                    )
                epoch_logs = copy.copy(logs)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):
                    # Create data_handler for evaluation and cache it.
                    if getattr(self, "_eval_data_handler", None) is None:
                        self._eval_data_handler = data_adapter.get_data_handler(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_epoch=validation_steps,
                            initial_epoch=0,
                            epochs=1,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing,
                            model=self,
                            steps_per_execution=self._steps_per_execution,
                        )
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks_list,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True,
                        _use_cached_eval_dataset=True,
                    )
                    val_logs = {"val_" + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks_list.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break

            # If eval_data_handler exists, delete it after all epochs are done.
            if getattr(self, "_eval_data_handler", None) is not None:
                del self._eval_data_handler
            callbacks_list.on_train_end(logs=training_logs)
            return self.history


class CustomDataHandler(data_adapter.DataHandler):
    """Handles iterating over epoch-level `tf.data.Iterator` objects."""

    def enumerate_epochs(self) -> Generator[Tuple[int, Iterator], None, None]:
        """Yields `(epoch, tf.data.Iterator)`."""
        # TODO
        #  we don't need this anymore once the fix from
        #  https://github.com/tensorflow/tensorflow/pull/45338
        #  has been ported over to keras and merged there

        # This code is adapted from
        # https://github.com/keras-team/keras/blob/r2.7/keras/engine/data_adapter.py#L1192

        with self._truncate_execution_to_epoch():
            data_iterator = iter(self._dataset)
            for epoch in range(self._initial_epoch, self._epochs):
                if self._insufficient_data:  # Set by `catch_stop_iteration`.
                    break
                # Adaption: update number of steps for epoch as we might have an
                # increasing batch size
                if self._adapter.should_recreate_iterator():
                    data_iterator = iter(self._dataset)
                    self._inferred_steps = len(self._adapter._keras_sequence)
                yield epoch, data_iterator
                self._adapter.on_epoch_end()
