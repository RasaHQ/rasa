import logging
import math
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Text,
    Tuple,
    Union,
    cast,
)

import numpy as np
import scipy.sparse
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from rasa.utils.tensorflow.constants import BALANCED, SEQUENCE
from rasa.utils.tensorflow.model_data import (
    Data,
    FeatureArray,
    FeatureSignature,
    RasaModelData,
)

logger = logging.getLogger(__name__)


class RasaDataGenerator(Sequence):
    """Abstract data generator."""

    def __init__(
        self,
        model_data: RasaModelData,
        batch_size: Union[int, List[int]],
        batch_strategy: Text = SEQUENCE,
        shuffle: bool = True,
    ):
        """Initializes the data generator.

        Args:
            model_data: The model data to use.
            batch_size: The batch size(s).
            batch_strategy: The batch strategy.
            shuffle: If 'True', data should be shuffled.
        """
        self.model_data = model_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_strategy = batch_strategy

    def __len__(self) -> int:
        """Number of batches in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Gets batch at position `index`.

        Arguments:
            index: position of the batch in the Sequence.

        Returns:
            A batch (tuple of input data and target data).
        """
        raise NotImplementedError

    def on_epoch_end(self) -> None:
        """Update the data after every epoch."""
        raise NotImplementedError

    def _shuffle_and_balance(self, batch_size: int) -> Data:
        data = self.model_data.data

        if self.shuffle:
            data = self.model_data.shuffled_data(data)

        if self.batch_strategy == BALANCED:
            data = self.model_data.balanced_data(data, batch_size, self.shuffle)

        # do not override self.model_data.data, because we need original data for
        # balancing on the next epoch
        return data

    @staticmethod
    def _create_default_array() -> np.ndarray:
        """Create a default empty array for missing features.

        Returns:
            A default empty array with shape (0, 1) and dtype float32.
        """
        return np.zeros((0, 1), dtype=np.float32)

    @staticmethod
    def prepare_batch(
        data: Data,
        start: Optional[int] = None,
        end: Optional[int] = None,
        tuple_sizes: Optional[Dict[Text, int]] = None,
    ) -> Tuple[np.ndarray, ...]:
        """Slices model data into batch using given start and end value.

        Args:
            data: The data to prepare.
            start: The start index of the batch
            end: The end index of the batch
            tuple_sizes: In case the feature is not present we propagate the batch with
              default arrays. Tuple sizes contains the number of how many default values
              to add for what kind of feature.

        Returns:
            The features of the batch.
        """
        batch_data = []

        for key, attribute_data in data.items():
            for sub_key, f_data in attribute_data.items():
                # add default arrays for not present values during processing
                if not f_data:
                    if tuple_sizes:
                        batch_data += [
                            RasaDataGenerator._create_default_array()
                        ] * tuple_sizes[key]
                    else:
                        batch_data.append(RasaDataGenerator._create_default_array())
                    continue

                for v in f_data:
                    if start is not None and end is not None:
                        _data = v[start:end]
                    elif start is not None:
                        _data = v[start:]
                    elif end is not None:
                        _data = v[:end]
                    else:
                        _data = v[:]

                    if cast(FeatureArray, _data).is_sparse:
                        batch_data.extend(
                            RasaDataGenerator._scipy_matrix_to_values(_data)
                        )
                    else:
                        batch_data.append(RasaDataGenerator._pad_dense_data(_data))

        # len of batch_data is equal to the number of keys in model data
        return tuple(batch_data)

    @staticmethod
    def _pad_dense_data(array_of_dense: FeatureArray) -> np.ndarray:
        """Pad data of different lengths.

        Sequential data is padded with zeros. Zeros are added to the end of data.

        Args:
            array_of_dense: The array to pad.

        Returns:
            The padded array.
        """
        if array_of_dense.number_of_dimensions == 4:
            return RasaDataGenerator._pad_4d_dense_data(array_of_dense)

        if array_of_dense[0].ndim < 2:
            # data doesn't contain a sequence
            return array_of_dense.astype(np.float32)

        data_size = len(array_of_dense)
        max_seq_len = max([x.shape[0] for x in array_of_dense])

        data_padded = np.zeros(
            [data_size, max_seq_len, array_of_dense[0].shape[-1]],
            dtype=array_of_dense[0].dtype,
        )
        for i in range(data_size):
            data_padded[i, : array_of_dense[i].shape[0], :] = array_of_dense[i]

        return data_padded.astype(np.float32)

    @staticmethod
    def _pad_4d_dense_data(feature_array: FeatureArray) -> np.ndarray:
        # in case of dialogue data we may have 4 dimensions
        # batch size x dialogue history length x sequence length x number of features

        # as transformers cannot handle 4D tensors pad and reshape the data
        # so that the resulting tensor is 3D
        # the shape is (sum of dialogue history length for all tensors in the
        # batch x max sequence length x number of features)
        # the original shape and the original dialogue length is passed on to the model
        # it can be used to transform the 3D tensor back into 4D

        # in order to create 4d tensor inputs, we created "fake" zero features
        # for nonexistent inputs. To save calculation we filter this features before
        # input to tf methods.
        number_of_features = feature_array[0][0].shape[-1]
        array_of_array_of_dense = RasaDataGenerator._filter_out_fake_inputs(
            feature_array
        )
        if not array_of_array_of_dense:
            # return empty 3d array with appropriate last dims
            return np.zeros((0, 0, number_of_features), dtype=np.float32)

        combined_dialogue_len = sum(
            len(array_of_dense) for array_of_dense in array_of_array_of_dense
        )
        max_seq_len = max(
            [
                x.shape[0]
                for array_of_dense in array_of_array_of_dense
                for x in array_of_dense
            ]
        )

        data_padded = np.zeros(
            [combined_dialogue_len, max_seq_len, number_of_features],
            dtype=array_of_array_of_dense[0][0].dtype,
        )

        current_sum_dialogue_len = 0
        for i, array_of_dense in enumerate(array_of_array_of_dense):
            for j, dense in enumerate(array_of_dense):
                data_padded[current_sum_dialogue_len + j, : dense.shape[0], :] = dense
            current_sum_dialogue_len += len(array_of_dense)

        return data_padded.astype(np.float32)

    @staticmethod
    def _scipy_matrix_to_values(array_of_sparse: FeatureArray) -> List[np.ndarray]:
        """Convert a scipy matrix into indices, data, and shape.

        Args:
            array_of_sparse: The sparse data array.

        Returns:
            A list of dense numpy arrays representing the sparse data.
        """
        if array_of_sparse.number_of_dimensions == 4:
            return RasaDataGenerator._4d_scipy_matrix_to_values(array_of_sparse)

        # we need to make sure that the matrices are coo_matrices otherwise the
        # transformation does not work (e.g. you cannot access x.row, x.col)
        if not isinstance(array_of_sparse[0], scipy.sparse.coo_matrix):
            array_of_sparse = [x.tocoo() for x in array_of_sparse]  # type: ignore[assignment]

        max_seq_len = max([x.shape[0] for x in array_of_sparse])

        # get the indices of values
        indices = np.hstack(
            [
                np.vstack([i * np.ones_like(x.row), x.row, x.col])
                for i, x in enumerate(array_of_sparse)
            ]
        ).T

        data = np.hstack([x.data for x in array_of_sparse])

        number_of_features = array_of_sparse[0].shape[-1]
        shape = np.array((len(array_of_sparse), max_seq_len, number_of_features))

        return [
            indices.astype(np.int64),
            data.astype(np.float32),
            shape.astype(np.int64),
        ]

    @staticmethod
    def _4d_scipy_matrix_to_values(feature_array: FeatureArray) -> List[np.ndarray]:
        # in case of dialogue data we may have 4 dimensions
        # batch size x dialogue history length x sequence length x number of features

        # transformers cannot handle 4D tensors, therefore pad and reshape the data
        # so that the resulting tensor is 3D
        # the shape is (sum of dialogue history length for all tensors in the
        # batch x max sequence length x number of features)
        # the original shape and the original dialogue length is passed on to the model
        # it can be used to transform the 3D tensor back into 4D

        # in order to create 4d tensor inputs, we created "fake" zero features
        # for nonexistent inputs. To save calculation we filter this features before
        # input to tf methods.
        number_of_features = feature_array[0][0].shape[-1]
        array_of_array_of_sparse = RasaDataGenerator._filter_out_fake_inputs(
            feature_array
        )
        if not array_of_array_of_sparse:
            # create empty array with appropriate last dims
            return [
                np.empty((0, 3), dtype=np.int64),
                np.array([], dtype=np.float32),
                np.array([0, 0, number_of_features], dtype=np.int64),
            ]

        # we need to make sure that the matrices are coo_matrices otherwise the
        # transformation does not work (e.g. you cannot access x.row, x.col)
        if not isinstance(array_of_array_of_sparse[0][0], scipy.sparse.coo_matrix):
            array_of_array_of_sparse = [
                [
                    x.tocoo() if isinstance(x, scipy.sparse.spmatrix) else x
                    for x in array_of_sparse
                ]
                for array_of_sparse in array_of_array_of_sparse
            ]

        dialogue_len = [
            len(array_of_sparse) for array_of_sparse in array_of_array_of_sparse
        ]
        combined_dialogue_len = sum(dialogue_len)
        max_seq_len = max(
            [
                x.shape[0]
                for array_of_sparse in array_of_array_of_sparse
                for x in array_of_sparse
            ]
        )
        # get the indices of values
        indices = np.hstack(
            [
                np.vstack(
                    [sum(dialogue_len[:i]) + j * np.ones_like(x.row), x.row, x.col]
                )
                for i, array_of_sparse in enumerate(array_of_array_of_sparse)
                for j, x in enumerate(array_of_sparse)
            ]
        ).T

        data = np.hstack(
            [
                x.data
                for array_of_sparse in array_of_array_of_sparse
                for x in array_of_sparse
            ]
        )

        shape = np.array((combined_dialogue_len, max_seq_len, number_of_features))

        return [
            indices.astype(np.int64),
            data.astype(np.float32),
            shape.astype(np.int64),
        ]

    @staticmethod
    def _filter_out_fake_inputs(
        array_of_array_of_features: FeatureArray,
    ) -> Union[List[List[np.ndarray]], List[List[scipy.sparse.spmatrix]]]:
        return list(
            filter(
                # filter empty lists created by another filter
                lambda x: len(x) > 0,
                [
                    # filter all the "fake" inputs, we know the input is "fake",
                    # when sequence dimension is `0`
                    list(filter(lambda x: x.shape[0] > 0, array_of_features))
                    for array_of_features in array_of_array_of_features
                ],
            )
        )


class RasaBatchDataGenerator(RasaDataGenerator):
    """Data generator with an optional increasing batch size."""

    def __init__(
        self,
        model_data: RasaModelData,
        batch_size: Union[List[int], int],
        epochs: int = 1,
        batch_strategy: Text = SEQUENCE,
        shuffle: bool = True,
        drop_small_last_batch: bool = False,
    ):
        """Initializes the increasing batch size data generator.

        Args:
            model_data: The model data to use.
            batch_size: The batch size.
            epochs: The total number of epochs.
            batch_strategy: The batch strategy.
            shuffle: If 'True', data will be shuffled.
            drop_small_last_batch: if 'True', the last batch in an epoch will be dropped
                if it has less examples than half the batch size
        """
        super().__init__(model_data, batch_size, batch_strategy, shuffle)

        if isinstance(batch_size, list):
            logger.debug(
                "The provided batch size is a list, this data generator will use a "
                "linear increasing batch size."
            )

        self._epochs = epochs
        # we use `on_epoch_end` method to prepare data for the next epoch
        # set current epoch to `-1`, so that `on_epoch_end` will increase it to `0`
        self._current_epoch = -1
        # actual batch size will be set inside `on_epoch_end`
        self._current_batch_size = 0
        # create separate data variable that will store modified data for each batch
        self._data: Data = {}
        self.drop_small_last_batch = drop_small_last_batch
        self.on_epoch_end()

    def __len__(self) -> int:
        """Number of batches in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        # data was rebalanced, so need to recalculate number of examples
        num_examples = self.model_data.number_of_examples(self._data)
        batch_size = self._current_batch_size
        if self.drop_small_last_batch:
            # keep last batch only if it has at least half a batch size of examples
            last_batch_half_full = num_examples % batch_size >= math.ceil(
                batch_size / 2
            )
            num_batches = num_examples // batch_size + int(last_batch_half_full)
            # Return at least 1 if there is an example
            return max(num_batches, int(num_examples > 0))
        else:
            return num_examples // batch_size + int(num_examples % batch_size > 0)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Gets batch at position `index`.

        Arguments:
            index: position of the batch in the Sequence.

        Returns:
            A batch (tuple of input data and target data).
        """
        start = index * self._current_batch_size
        end = start + self._current_batch_size

        # return input and target data, as our target data is inside the input
        # data return default array for the target data
        return self.prepare_batch(
            self._data, start, end
        ), RasaDataGenerator._create_default_array()

    def on_epoch_end(self) -> None:
        """Update the data after every epoch."""
        self._current_epoch += 1
        self._current_batch_size = self._linearly_increasing_batch_size()
        self._data = self._shuffle_and_balance(self._current_batch_size)

    def _linearly_increasing_batch_size(self) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489.

        Returns:
            The batch size to use in this epoch.
        """
        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self._epochs > 1:
            return int(
                self.batch_size[0]
                + self._current_epoch
                * (self.batch_size[1] - self.batch_size[0])
                / (self._epochs - 1)
            )
        else:
            return int(self.batch_size[0])


def tf_data_generator_from_rasa_data_generator(
    generator: RasaDataGenerator,
) -> tf.data.Dataset:
    """Convert a RasaDataGenerator to a tf.data.Dataset.

    This adapter bridges the gap between Rasa's custom data generator (which produces
    variable-length batches) and TensorFlow's stricter shape validation in 2.19+.

    It performs two main functions:

    1.  **Defines an Explicit Signature**: It constructs a `tf.TensorSpec` signature
        with `None` dimensions for batch size and sequence length. This prevents
        TensorFlow from inferring fixed shapes from the first batch.
    2.  **Flattens Data Structure**: It transforms the nested dictionary output of
        `RasaDataGenerator` into a flat tuple of tensors, as expected by `RasaModel`.
        *Note*: Previously, when passing a Keras Sequence directly to `model.fit`, Keras
        handled this flattening implicitly by mapping dictionary keys to model inputs.
        With `tf.data.Dataset`, we must perform this flattening explicitly to match
        the model's input signature.

    Returns:
        A `tf.data.Dataset` that yields tuples of `(inputs, targets)` where:
        - `inputs` is a flat tuple of tensors matching the model's signature.
        - `targets` is a tensor (or tuple of tensors) for labels.
    """
    signature = generator.model_data.get_signature()

    flat_specs = _determine_flat_specs(generator)
    target_spec = _determine_target_spec(generator)
    output_signature = (tuple(flat_specs), target_spec)

    def generator_func() -> Iterator[Tuple[Any, Any]]:
        # This function is called by TensorFlow once per epoch (or iteration).
        # It iterates over the RasaDataGenerator, yielding batches.
        for i in range(len(generator)):
            batch = generator[i]
            inputs = batch[0]
            batch_targets = batch[1]

            # RasaBatchDataGenerator.prepare_batch returns a tuple of arrays.
            # We yield it as is, matching the flat signature.
            if isinstance(inputs, (list, tuple)):
                yield tuple(inputs), batch_targets
            elif isinstance(inputs, dict):
                # Legacy/Fallback: If it happens to be a dict, we flatten it
                flat_inputs = []
                for key, attribute_data in signature.items():
                    for sub_key, features in attribute_data.items():
                        arrays = inputs[key][sub_key]
                        for array in arrays:
                            flat_inputs.append(array)
                yield tuple(flat_inputs), batch_targets
            else:
                # Fallback
                yield inputs, batch_targets

        # Trigger shuffling for the next epoch
        generator.on_epoch_end()

    return tf.data.Dataset.from_generator(
        generator_func, output_signature=output_signature
    )


def _determine_flat_specs(generator: RasaDataGenerator) -> List[tf.TensorSpec]:
    """Determine the flat list of TensorSpecs for the generator's inputs.

    Args:
        generator: The Rasa data generator.

    Returns:
        A list of TensorSpecs corresponding to the flattened input structure.
    """
    signature = generator.model_data.get_signature()

    # RasaModel expects a flat list of tensors as input, corresponding to the
    # flattened signature. We must construct the output_signature as a flat tuple.
    flat_specs = []

    for key, attribute_data in signature.items():
        for sub_key, features in attribute_data.items():
            for feature in features:
                if feature.is_sparse:
                    flat_specs.extend(_get_sparse_feature_specs())
                else:
                    flat_specs.extend(_get_dense_feature_specs(feature))
    return flat_specs


def _get_sparse_feature_specs() -> List[tf.TensorSpec]:
    """Get TensorSpecs for a sparse feature (e.g., Bag-of-Words).

    Returns:
        List of TensorSpecs for indices, values, and shape.
    """
    specs = []
    # RasaModel expects decomposed sparse tensor parts: indices, values, shape

    # 1. Indices: (n_elements, rank)
    #    - n_elements: Total number of non-zero values in the batch (variable).
    #    - rank: Number of dimensions (fixed at 3 for Rasa sparse features).
    specs.append(tf.TensorSpec(shape=(None, 3), dtype=tf.int64))

    # 2. Values: (n_elements,)
    #    - The actual values at the indices.
    specs.append(tf.TensorSpec(shape=(None,), dtype=tf.float32))

    # 3. Dense Shape: (rank,) -> (3,)
    #    - The logical shape of the sparse tensor: (batch_size, max_seq_len, n_features)
    specs.append(tf.TensorSpec(shape=(3,), dtype=tf.int64))

    return specs


def _get_dense_feature_specs(feature: FeatureSignature) -> List[tf.TensorSpec]:
    """Get TensorSpecs for a dense feature.

    Args:
        feature: The feature array to determine specs for.

    Returns:
        List containing the single TensorSpec for this dense feature.
    """
    if feature.number_of_dimensions == 1:
        # 1D Feature (e.g., Labels)
        # Shape: (batch_size,)
        shape = [None]
    else:
        # Dense Feature
        # Shape depends on dimensionality:
        # - 2D: (batch_size, n_features) -> [None, units]
        # - 3D: (batch_size, max_seq_len, n_features) -> [None, None, units]

        # We use None for all dimensions except the last one (units)
        shape = [None] * (feature.number_of_dimensions - 1)

        if feature.number_of_dimensions == 4:
            # 4D features (Dialogue) are flattened to 3D in `prepare_batch`.
            # Source Shape: (batch_size, dialog_len, seq_len, n_features)
            # Yielded Shape: (flattened_batch, max_seq_len, n_features)
            # where flattened_batch = sum(dialog_len) over the batch.

            # We override the shape to match the yielded 3D data:
            shape = [None, None]

        shape.append(feature.units)

    return [tf.TensorSpec(shape=shape, dtype=tf.float32)]


def _determine_target_spec(
    generator: RasaDataGenerator,
) -> Union[tf.TensorSpec, Tuple[tf.TensorSpec, ...]]:
    """Determine the target Tensor spec from the first batch of the generator.

    Args:
        generator: The Rasa data generator.

    Returns:
        The TensorSpec (or tuple of specs) for the targets.
    """
    target_spec = None
    if isinstance(generator, RasaBatchDataGenerator):
        # Optimization: RasaBatchDataGenerator always returns default empty
        # targets (shape (0, 1)).
        # We can skip inspecting the first batch in this common case to avoid
        # performance cost of computing the batch twice.
        target_spec = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    else:
        # For other generators, we must inspect the first batch to determine target
        # shape.
        # Note: Accessing generator[0] should be side-effect free (idempotent) regarding
        # the generator's internal state (e.g., shuffling), as __getitem__ is typically
        # read-only.
        try:
            first_batch = generator[0]
            targets = first_batch[1]

            if isinstance(targets, np.ndarray):
                # Targets shape is (batch_size, n_targets)
                # We set the batch dimension (0) to None
                target_shape = [None] + list(targets.shape[1:])
                target_spec = tf.TensorSpec(shape=target_shape, dtype=targets.dtype)
            elif isinstance(targets, list):
                # If targets is a list of arrays
                target_specs = []
                for t in targets:
                    # Shape: (batch_size, n_targets)
                    target_shape = [None] + list(t.shape[1:])
                    target_specs.append(
                        tf.TensorSpec(shape=target_shape, dtype=t.dtype)
                    )
                target_spec = tuple(target_specs)
        except Exception:
            # Fallback if generator is empty or fails
            pass

    if target_spec is None:
        # Fallback or empty targets
        # Shape: (batch_size, 1)
        target_spec = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)

    return target_spec
