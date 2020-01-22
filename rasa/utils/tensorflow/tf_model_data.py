import numpy as np
import scipy.sparse
import tensorflow as tf

from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Text, List, Tuple, Any, Union, Generator, NamedTuple
from collections import defaultdict


Data = Optional[Dict[Text, List[np.ndarray]]]

class FeatureSignature(NamedTuple):
    is_sparse: bool
    shape: List[int]


class RasaModelData:
    def __init__(
        self,
        label_key: Optional[Text] = None,
        data: Data = None,
    ):
        self.data = data or {}
        self.label_key = label_key or ""
        # will be updated when features are added
        self.num_examples = self.get_number_of_examples()

    def get(self, key: Text) -> List[np.ndarray]:
        return self.data[key]

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def keys(self):
        return self.data.keys()

    def feature_not_exists(self, key: Text) -> bool:
        return key not in self.data or not self.data[key]

    def get_number_of_examples(self) -> int:
        """Obtain number of examples in session data.

        Raise a ValueError if number of examples differ for different data in
        session data.
        """
        if not self.data:
            return 0

        example_lengths = [v.shape[0] for values in self.data.values() for v in values]

        # check if number of examples is the same for all values
        if not all(length == example_lengths[0] for length in example_lengths):
            raise ValueError(
                f"Number of examples differs for keys '{self.data.keys()}'. Number of "
                f"examples should be the same for all data."
            )

        return example_lengths[0]

    def get_feature_dimension(self, key: Text) -> int:
        """Get the feature dimension of the given key."""

        number_of_features = 0
        for data in self.data[key]:
            if data.size > 0:
                number_of_features += data[0].shape[-1]

        return number_of_features

    def split(
        self, number_of_test_examples: int, random_seed: int
    ) -> Tuple["RasaModelData", "RasaModelData"]:
        """Create random hold out test set using stratified split."""

        self._check_label_key(self.label_key)

        label_ids = self._create_label_ids(self.data[self.label_key][0])
        label_counts = dict(zip(*np.unique(label_ids, return_counts=True, axis=0)))

        self._check_train_test_sizes(number_of_test_examples, label_counts)

        counts = np.array([label_counts[label] for label in label_ids])
        multi_values = [v[counts > 1] for values in self.data.values() for v in values]
        solo_values = [v[counts == 1] for values in self.data.values() for v in values]

        output_values = train_test_split(
            *multi_values,
            test_size=number_of_test_examples,
            random_state=random_seed,
            stratify=label_ids[counts > 1],
        )

        return self._convert_train_test_split(output_values, solo_values)

    def add_features(self, key: Text, features: List[np.ndarray]):
        """Add list of features to data under specified key."""

        if not features:
            return

        if key in self.data:
            raise ValueError(f"Key '{key}' already exists in RasaModelData.")

        self.data[key] = []

        for data in features:
            if data.size > 0:
                self.data[key].append(data)

        if not self.data[key]:
            del self.data[key]

        # update number of examples
        self.num_examples = self.get_number_of_examples()

    def add_mask(self, key: Text, from_key: Text):
        """Calculate mask for given key and put it under specified key."""

        if not self.data.get(from_key):
            return

        self.data[key] = []

        for data in self.data[from_key]:
            if data.size > 0:
                # explicitly add last dimension to mask
                # to track correctly dynamic sequences
                mask = np.array([np.ones((x.shape[0], 1)) for x in data])
                self.data[key].append(mask)
                break

    def get_signature(self) -> Dict[Text, List[FeatureSignature]]:
        """Get signature of RasaModelData.

        Signature stores the shape and whether features are sparse or not for every
        key."""

        return {
            key: [
                FeatureSignature(
                    True if isinstance(v[0], scipy.sparse.spmatrix) else False,
                    v[0].shape,
                )
                for v in values
            ]
            for key, values in self.data.items()
        }

    def shuffled_data(self, data: Data) -> Data:
        """Shuffle session data."""

        ids = np.random.permutation(self.num_examples)
        return self._data_for_ids(data, ids)

    def balanced_data(self, data: Data, batch_size: int, shuffle: bool) -> Data:
        """Mix session data to account for class imbalance.

        This batching strategy puts rare classes approximately in every other batch,
        by repeating them. Mimics stratified batching, but also takes into account
        that more populated classes should appear more often.
        """

        if self.label_key not in data or len(data[self.label_key]) > 1:
            raise ValueError(f"Key '{self.label_key}' not in RasaModelData.")

        label_ids = self._create_label_ids(data[self.label_key][0])

        unique_label_ids, counts_label_ids = np.unique(
            label_ids, return_counts=True, axis=0
        )
        num_label_ids = len(unique_label_ids)

        # need to call every time, so that the data is shuffled inside each class
        label_data = self._split_by_label_ids(data, label_ids, unique_label_ids)

        data_idx = [0] * num_label_ids
        num_data_cycles = [0] * num_label_ids
        skipped = [False] * num_label_ids

        new_data = defaultdict(list)

        while min(num_data_cycles) == 0:
            if shuffle:
                indices_of_labels = np.random.permutation(num_label_ids)
            else:
                indices_of_labels = range(num_label_ids)

            for index in indices_of_labels:
                if num_data_cycles[index] > 0 and not skipped[index]:
                    skipped[index] = True
                    continue
                else:
                    skipped[index] = False

                index_batch_size = (
                    int(counts_label_ids[index] / self.num_examples * batch_size) + 1
                )

                for k, values in label_data[index].items():
                    for i, v in enumerate(values):
                        if len(new_data[k]) < i + 1:
                            new_data[k].append([])
                        new_data[k][i].append(
                            v[data_idx[index] : data_idx[index] + index_batch_size]
                        )

                data_idx[index] += index_batch_size
                if data_idx[index] >= counts_label_ids[index]:
                    num_data_cycles[index] += 1
                    data_idx[index] = 0

                if min(num_data_cycles) > 0:
                    break

        final_data = defaultdict(list)
        for k, values in new_data.items():
            for v in values:
                final_data[k].append(np.concatenate(np.array(v)))

        return final_data

    def prepare_batch(
        self,
        data: Optional[Data] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        tuple_sizes: Optional[Dict[Text, int]] = None,
    ) -> Tuple[Optional[np.ndarray]]:
        """Slices session data into batch using given start and end value."""

        if not data:
            data = self.data

        batch_data = []

        for key, values in data.items():
            # add None for not present values during processing
            if not values:
                if tuple_sizes:
                    batch_data += [None] * tuple_sizes[key]
                else:
                    batch_data.append(None)
                continue

            for v in values:
                if start is not None and end is not None:
                    _data = v[start:end]
                elif start is not None:
                    _data = v[start:]
                elif end is not None:
                    _data = v[:end]
                else:
                    _data = v[:]

                if isinstance(_data[0], scipy.sparse.spmatrix):
                    batch_data.extend(self._scipy_matrix_to_values(_data))
                else:
                    batch_data.append(self._pad_dense_data(_data))

        # len of batch_data is equal to the number of keys in session data
        return tuple(batch_data)

    def batch_tuple_sizes(self) -> Dict[Text, int]:

        # save the amount of placeholders attributed to session data keys
        tuple_sizes = defaultdict(int)

        idx = 0
        for k, values in self.data.items():
            tuple_sizes[k] = 0
            for v in values:
                if isinstance(v[0], scipy.sparse.spmatrix):
                    tuple_sizes[k] += 3
                    idx += 3
                else:
                    tuple_sizes[k] += 1
                    idx += 1

        return tuple_sizes

    def as_tf_dataset(
        self,
        batch_size: Union["tf.Tensor", int],
        batch_strategy: Text = "sequence",
        shuffle: bool = False,
    ) -> "tf.data.Dataset":
        """Create tf dataset."""

        shapes, types = self._get_shapes_types()

        return tf.data.Dataset.from_generator(
            lambda batch_size_: self._gen_batch(batch_size_, batch_strategy, shuffle),
            output_types=types,
            output_shapes=shapes,
            args=([batch_size]),
        )

    def _get_shapes_types(self) -> Tuple:
        """Extract shapes and types from session data."""

        types = []
        shapes = []

        def append_shape(features: np.ndarray):
            if isinstance(features[0], scipy.sparse.spmatrix):
                # scipy matrix is converted into indices, data, shape
                shapes.append((None, features[0].ndim + 1))
                shapes.append((None,))
                shapes.append((features[0].ndim + 1))
            elif features[0].ndim == 0:
                shapes.append((None,))
            elif features[0].ndim == 1:
                shapes.append((None, features[0].shape[-1]))
            else:
                shapes.append((None, None, features[0].shape[-1]))

        def append_type(features: np.ndarray):
            if isinstance(features[0], scipy.sparse.spmatrix):
                # scipy matrix is converted into indices, data, shape
                types.append(tf.int64)
                types.append(tf.float32)
                types.append(tf.int64)
            else:
                types.append(tf.float32)

        for values in self.data.values():
            for v in values:
                append_shape(v)
                append_type(v)

        return tuple(shapes), tuple(types)

    def _gen_batch(
        self, batch_size: int, batch_strategy: Text = "sequence", shuffle: bool = False
    ) -> Generator[Tuple, None, None]:
        """Generate batches."""

        data = self.data

        if shuffle:
            data = self.shuffled_data(data)

        if batch_strategy == "balanced":
            data = self.balanced_data(data, batch_size, shuffle)

        num_batches = self.num_examples // batch_size + int(self.num_examples % batch_size > 0)

        for batch_num in range(num_batches):
            start = batch_num * batch_size
            end = start + batch_size

            yield self.prepare_batch(data, start, end)

    def _check_train_test_sizes(
        self, number_of_test_examples: int, label_counts: Dict[Any, int]
    ):
        """Check whether the test data set is too large or too small."""

        if number_of_test_examples >= self.num_examples - len(label_counts):
            raise ValueError(
                f"Test set of {number_of_test_examples} is too large. Remaining "
                f"train set should be at least equal to number of classes "
                f"{len(label_counts)}."
            )
        elif number_of_test_examples < len(label_counts):
            raise ValueError(
                f"Test set of {number_of_test_examples} is too small. It should "
                f"be at least equal to number of classes {label_counts}."
            )

    @staticmethod
    def _data_for_ids(data: Data, ids: np.ndarray) -> Dict[Text, List[np.ndarray]]:
        """Filter session data by ids."""

        new_data = defaultdict(list)
        for k, values in data.items():
            for v in values:
                new_data[k].append(v[ids])
        return new_data

    def _split_by_label_ids(
        self, data: Data, label_ids: "np.ndarray", unique_label_ids: "np.ndarray"
    ) -> List["RasaModelData"]:
        """Reorganize session data into a list of session data with the same labels."""

        label_data = []
        for label_id in unique_label_ids:
            ids = label_ids == label_id
            label_data.append(RasaModelData(self.label_key, self._data_for_ids(data, ids)))
        return label_data

    def _check_label_key(self, label_key: Text):
        if label_key not in self.data or len(self.data[label_key]) > 1:
            raise ValueError(f"Key '{label_key}' not in RasaModelData.")

    def _convert_train_test_split(
        self, output_values: List[Any], solo_values: List[Any]
    ) -> Tuple["RasaModelData", "RasaModelData"]:
        """Convert the output of sklearn.model_selection.train_test_split into train and
        eval session data."""

        data_train = defaultdict(list)
        data_val = defaultdict(list)

        # output_values = x_train, x_val, y_train, y_val, z_train, z_val, etc.
        # order is kept, e.g. same order as session data keys

        # train datasets have an even index
        index = 0
        for key, values in self.data.items():
            for _ in range(len(values)):
                data_train[key].append(
                    self._combine_features(output_values[index * 2], solo_values[index])
                )
                index += 1

        # val datasets have an odd index
        index = 0
        for key, values in self.data.items():
            for _ in range(len(values)):
                data_val[key].append(output_values[(index * 2) + 1])
                index += 1

        return (
            RasaModelData(self.label_key, data_train),
            RasaModelData(self.label_key, data_val),
        )

    @staticmethod
    def _combine_features(
        feature_1: Union[np.ndarray, scipy.sparse.spmatrix],
        feature_2: Union[np.ndarray, scipy.sparse.spmatrix],
    ) -> Union[np.ndarray, scipy.sparse.spmatrix]:
        """Concatenate features."""

        if isinstance(feature_1, scipy.sparse.spmatrix) and isinstance(
            feature_2, scipy.sparse.spmatrix
        ):
            if feature_2.shape[0] == 0:
                return feature_1
            if feature_1.shape[0] == 0:
                return feature_2
            return scipy.sparse.vstack([feature_1, feature_2])

        return np.concatenate([feature_1, feature_2])

    @staticmethod
    def _create_label_ids(label_ids: np.ndarray) -> np.ndarray:
        """Convert various size label_ids into single dim array.

        For multi-label y, map each distinct row to a string representation
        using join because str(row) uses an ellipsis if len(row) > 1000.
        Idea taken from sklearn's stratify split.
        """

        if label_ids.ndim == 1:
            return label_ids

        if label_ids.ndim == 2 and label_ids.shape[-1] == 1:
            return label_ids[:, 0]

        if label_ids.ndim == 2:
            return np.array([" ".join(row.astype("str")) for row in label_ids])

        if label_ids.ndim == 3 and label_ids.shape[-1] == 1:
            return np.array([" ".join(row.astype("str")) for row in label_ids[:, :, 0]])

        raise ValueError("Unsupported label_ids dimensions")

    @staticmethod
    def _pad_dense_data(array_of_dense: np.ndarray) -> np.ndarray:
        """Pad data of different lengths.

        Sequential data is padded with zeros. Zeros are added to the end of data.
        """

        if array_of_dense[0].ndim < 2:
            # data doesn't contain a sequence
            return array_of_dense

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
    def _scipy_matrix_to_values(array_of_sparse: np.ndarray) -> List[np.ndarray]:
        """Convert a scipy matrix into inidces, data, and shape."""

        if not isinstance(array_of_sparse[0], scipy.sparse.coo_matrix):
            array_of_sparse = [x.tocoo() for x in array_of_sparse]

        max_seq_len = max([x.shape[0] for x in array_of_sparse])

        indices = np.hstack(
            [
                np.vstack([i * np.ones_like(x.row), x.row, x.col])
                for i, x in enumerate(array_of_sparse)
            ]
        ).T
        data = np.hstack([x.data for x in array_of_sparse])

        shape = np.array(
            (len(array_of_sparse), max_seq_len, array_of_sparse[0].shape[-1])
        )

        return [
            indices.astype(np.int64),
            data.astype(np.float32),
            shape.astype(np.int64),
        ]
