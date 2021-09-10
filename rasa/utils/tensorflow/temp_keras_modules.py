import copy
from typing import (
    List,
    Dict,
    Union,
    Optional,
    Any,
    Generator,
    Tuple,
    Iterator,
)

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.callbacks import Callback, History
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.eager import context
from tensorflow.python.keras.engine.data_adapter import DataHandler


# noinspection PyMethodOverriding
class TmpKerasModel(tf.keras.models.Model):
    """Temporary solution. Keras model that uses a custom data adapter inside fit."""

    # TODO
    #  we don't need this anymore once
    #  https://github.com/tensorflow/tensorflow/pull/45338
    #  is merged and released

    # This code is adapted from
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/engine/training.py#L824-L1146

    @training.enable_multi_worker
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
            x: Input data.
            y: Target data.
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train the model.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per
                     epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
            validation_split: Fraction of the training data to be used as validation
                              data.
            validation_data: Data on which to evaluate the loss and any model metrics
                             at the end of each epoch.
            shuffle: whether to shuffle the training data before each epoch
            class_weight: Optional dictionary mapping class indices (integers)
                          to a weight (float) value, used for weighting the loss
                          function (during training only).
            sample_weight: Optional Numpy array of weights for
                           the training samples, used for weighting the loss function
                           (during training only).
            initial_epoch: Epoch at which to start training
            steps_per_epoch: Total number of steps (batches of samples)
                             before declaring one epoch finished and starting the
                             next epoch.
            validation_steps: Total number of steps (batches of
                              samples) to draw before stopping when performing
                              validation at the end of every epoch.
            validation_batch_size: Number of samples per validation batch.
            validation_freq: specifies how many training epochs to run before a
                             new validation run is performed
            max_queue_size: Maximum size for the generator queue.
            workers: Maximum number of processes to spin up
                     when using process-based threading.
            use_multiprocessing: If `True`, use process-based threading.

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        Raises:
            RuntimeError: 1. If the model was never compiled or,
            2. If `model.fit` is  wrapped in `tf.function`.

            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        training._keras_api_gauge.get_cell("fit").set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph("Model", "fit")
        self._assert_compile_was_called()
        self._check_call_args("fit")
        training._disallow_inside_tf_function("fit")

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

        with self.distribute_strategy.scope(), (
            training_utils.RespectCompiledTrainableState(self)
        ):
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            # Use our own custom data handler to handle increasing batch size
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
            if not isinstance(callbacks, training.callbacks_module.CallbackList):
                callbacks = training.callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps,
                )

            self.stop_training = False
            train_function = self.make_train_function()
            self._train_counter.assign(0)
            callbacks.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            data_handler._initial_epoch = self._maybe_load_initial_epoch_from_ckpt(  # pylint: disable=protected-access # noqa: E501
                initial_epoch
            )
            for epoch, iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        with training.trace.Trace(
                            "TraceContext",
                            graph_type="train",
                            epoch_num=epoch,
                            step_num=step,
                            batch_size=batch_size,
                        ):
                            callbacks.on_train_batch_begin(step)
                            tmp_logs = train_function(iterator)
                            if data_handler.should_sync:
                                context.async_wait()
                            logs = tmp_logs  # No error, now safe to assign to logs.
                            end_step = step + data_handler.step_increment
                            callbacks.on_train_batch_end(end_step, logs)
                epoch_logs = copy.copy(logs)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):
                    # Create data_handler for evaluation and cache it.
                    if getattr(self, "_eval_data_handler", None) is None:
                        self._eval_data_handler = CustomDataHandler(
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
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True,
                    )
                    val_logs = {"val_" + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break

            # If _eval_data_handler exists, delete it after all epochs are done.
            if getattr(self, "_eval_data_handler", None) is not None:
                del self._eval_data_handler
            callbacks.on_train_end(logs=training_logs)
            return self.history


class CustomDataHandler(DataHandler):
    """Handles iterating over epoch-level `tf.data.Iterator` objects."""

    def enumerate_epochs(self) -> Generator[Tuple[int, Iterator], None, None]:
        """Yields `(epoch, tf.data.Iterator)`."""
        # TODO
        #  we don't need this anymore once
        #  https://github.com/tensorflow/tensorflow/pull/45338
        #  is merged and released

        # This code is adapted from
        # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/engine/data_adapter.py#L1135-L1145

        with self._truncate_execution_to_epoch():
            data_iterator = iter(self._dataset)
            for epoch in range(self._initial_epoch, self._epochs):
                if self._insufficient_data:  # Set by `catch_stop_iteration`.
                    break
                if self._adapter.should_recreate_iterator():
                    data_iterator = iter(self._dataset)
                    # update number of steps for epoch as we might have an increasing
                    # batch size
                    self._inferred_steps = len(self._adapter._keras_sequence)
                yield epoch, data_iterator
                self._adapter.on_epoch_end()
