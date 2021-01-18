import logging

import numpy as np
import scipy.sparse
import tensorflow as tf

from sklearn.model_selection import train_test_split
from typing import (
    Optional,
    Dict,
    Text,
    List,
    Tuple,
    Any,
    Union,
    Generator,
    NamedTuple,
    ItemsView,
)
from collections import defaultdict, OrderedDict
from rasa.utils.tensorflow.constants import BALANCED, SEQUENCE

logger = logging.getLogger(__name__)


class FeatureArray(np.ndarray):
    """Stores any kind of features ready to be used by a RasaModel.

    Next to the input numpy array of features, it also received the number of dimensions of the features.
    As our features can have 1 to 4 dimensions we might have different number of numpy arrays stacked.
    The number of dimensions helps us to figure out how to handle this particular feature array.
    Also, it is automatically determined whether the feature array is sparse or not and the number of units
    is determined as well.

    Subclassing np.array: https://numpy.org/doc/stable/user/basics.subclassing.html
    """

    def __new__(
        cls, input_array: np.ndarray, number_of_dimensions: int
    ) -> "FeatureArray":
        """Create and return a new object.  See help(type) for accurate signature."""
        FeatureArray._validate_number_of_dimensions(number_of_dimensions, input_array)

        feature_array = np.asarray(input_array).view(cls)

        if number_of_dimensions <= 2:
            feature_array.units = input_array.shape[-1]
            feature_array.is_sparse = isinstance(input_array[0], scipy.sparse.spmatrix)
        elif number_of_dimensions == 3:
            feature_array.units = input_array[0].shape[-1]
            feature_array.is_sparse = isinstance(input_array[0], scipy.sparse.spmatrix)
        elif number_of_dimensions == 4:
            feature_array.units = input_array[0][0].shape[-1]
            feature_array.is_sparse = isinstance(
                input_array[0][0], scipy.sparse.spmatrix
            )
        else:
            raise ValueError(
                f"Number of dimensions '{number_of_dimensions}' currently not supported."
            )

        feature_array.number_of_dimensions = number_of_dimensions

        return feature_array

    def __init__(self, input_array: Any, number_of_dimensions: int, **kwargs):
        """Initialize. FeatureArray.

        Needed in order to avoid 'Invalid keyword argument number_of_dimensions
        to function FeatureArray.__init__ '
        Args:
            input_array: the array that contains features
            number_of_dimensions: number of dimensions in input_array
        """
        super().__init__(**kwargs)
        self.number_of_dimensions = number_of_dimensions

    def __array_finalize__(self, obj: Any) -> None:
        """This method is called whenever the system internally allocates a new array from obj.

        Args:
            obj: A subclass (subtype) of ndarray.
        """
        if obj is None:
            return

        self.units = getattr(obj, "units", None)
        self.number_of_dimensions = getattr(obj, "number_of_dimensions", None)
        self.is_sparse = getattr(obj, "is_sparse", None)

        default_attributes = {
            "units": self.units,
            "number_of_dimensions": self.number_of_dimensions,
            "is_spare": self.is_sparse,
        }
        self.__dict__.update(default_attributes)

    # pytype: disable=attribute-error
    def __array_ufunc__(self, ufunc: Any, method: Text, *inputs, **kwargs) -> Any:
        """Overwrite this method as we are subclassing numpy array.

        Args:
            ufunc: The ufunc object that was called.
            method: A string indicating which Ufunc method was called
                    (one of "__call__", "reduce", "reduceat", "accumulate", "outer",
                    "inner").
            *inputs: A tuple of the input arguments to the ufunc.
            **kwargs: Any additional arguments

        Returns:
            The result of the operation.
        """
        f = {
            "reduce": ufunc.reduce,
            "accumulate": ufunc.accumulate,
            "reduceat": ufunc.reduceat,
            "outer": ufunc.outer,
            "at": ufunc.at,
            "__call__": ufunc,
        }
        # convert the inputs to np.ndarray to prevent recursion, call the function,
        # then cast it back as FeatureArray
        output = FeatureArray(
            f[method](*(i.view(np.ndarray) for i in inputs), **kwargs),
            number_of_dimensions=kwargs["number_of_dimensions"],
        )
        output.__dict__ = self.__dict__  # carry forward attributes
        return output

    def __reduce__(self) -> Tuple[Any, Any, Any]:
        """Needed in order to pickle this object.

        Returns:
            A tuple.
        """
        pickled_state = super(FeatureArray, self).__reduce__()
        new_state = pickled_state[2] + (
            self.number_of_dimensions,
            self.is_sparse,
            self.units,
        )
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state, **kwargs) -> None:
        """Sets the state.

        Args:
            state: The state argument must be a sequence that contains the following
                   elements version, shape, dtype, isFortan, rawdata.
            **kwargs: Any additional parameter
        """
        # Needed in order to load the object
        self.number_of_dimensions = state[-3]
        self.is_sparse = state[-2]
        self.units = state[-1]
        super(FeatureArray, self).__setstate__(state[0:-3], **kwargs)

    # pytype: enable=attribute-error

    @staticmethod
    def _validate_number_of_dimensions(
        number_of_dimensions: int, input_array: np.ndarray
    ) -> None:
        """Validates if the the input array has given number of dimensions.

        Args:
            number_of_dimensions: number of dimensions
            input_array: input array

        Raises: ValueError in case the dimensions do not match
        """
        _sub_array = input_array
        dim = 0
        # Go number_of_dimensions into the given input_array
        for i in range(1, number_of_dimensions + 1):
            _sub_array = _sub_array[0]
            if isinstance(_sub_array, scipy.sparse.spmatrix):
                dim = i
                break
            if isinstance(_sub_array, np.ndarray) and _sub_array.shape[0] == 0:
                # sequence dimension is 0, we are dealing with "fake" features
                dim = i
                break

        # If the resulting sub_array is sparse, the remaining number of dimensions
        # should be at least 2
        if isinstance(_sub_array, scipy.sparse.spmatrix):
            if dim > 2:
                raise ValueError(
                    f"Given number of dimensions '{number_of_dimensions}' does not "
                    f"match dimensions of given input array: {input_array}."
                )
        elif isinstance(_sub_array, np.ndarray) and _sub_array.shape[0] == 0:
            # sequence dimension is 0, we are dealing with "fake" features,
            # but they should be of dim 2
            if dim > 2:
                raise ValueError(
                    f"Given number of dimensions '{number_of_dimensions}' does not "
                    f"match dimensions of given input array: {input_array}."
                )
        # If the resulting sub_array is dense, the sub_array should be a single number
        elif not np.issubdtype(type(_sub_array), np.integer) and not isinstance(
            _sub_array, (np.float32, np.float64)
        ):
            raise ValueError(
                f"Given number of dimensions '{number_of_dimensions}' does not match "
                f"dimensions of given input array: {input_array}."
            )

    def get_shape_type_info(
        self,
    ) -> Tuple[
        List[
            Union[
                int,
                Tuple[None],
                Tuple[None, int],
                Tuple[None, None, int],
                Tuple[None, None, None, int],
            ]
        ],
        List[int],
    ]:
        """Returns shapes and types needed to convert this feature array into tensors.

        Returns:
            A list of shape tuples.
            A list of type tuples.
        """
        if self.is_sparse:
            # 4D tensors were converted into 3D tensors during padding
            number_of_dimensions = (
                self.number_of_dimensions if self.number_of_dimensions != 4 else 3
            )
            # scipy matrix is converted into indices, data, shape
            return (
                [(None, number_of_dimensions), (None,), (number_of_dimensions)],
                [tf.int64, tf.float32, tf.int64],
            )

        if self.number_of_dimensions == 1:
            return [(None,)], [tf.float32]

        if self.number_of_dimensions == 2:
            return [(None, self.units)], [tf.float32]

        if self.number_of_dimensions == 3:
            return [(None, None, self.units)], [tf.float32]

        if self.number_of_dimensions == 4:
            # 4D tensors were converted into 3D tensors during padding
            return [(None, None, self.units)], [tf.float32]

        return [], []


class FeatureSignature(NamedTuple):
    """Signature of feature arrays.

    Stores the number of units, the type (sparse vs dense), and the number of
    dimensions of features.
    """

    is_sparse: bool
    units: Optional[int]
    number_of_dimensions: int


# Mapping of attribute name and feature name to a list of feature arrays representing
# the actual features
# For example:
# "text" -> { "sentence": [
#   "feature array containing dense features for every training example",
#   "feature array containing sparse features for every training example"
# ]}
Data = Dict[Text, Dict[Text, List[FeatureArray]]]


class RasaModelData:
    """Data object used for all RasaModels.

    It contains all features needed to train the models.
    'data' is a mapping of attribute name, e.g. TEXT, INTENT, etc., and feature name,
    e.g. SENTENCE, SEQUENCE, etc., to a list of feature arrays representing the actual
    features.
    'label_key' and 'label_sub_key' point to the labels inside 'data'. For
    example, if your intent labels are stored under INTENT -> IDS, 'label_key' would
    be "INTENT" and 'label_sub_key' would be "IDS".
    """

    def __init__(
        self,
        label_key: Optional[Text] = None,
        label_sub_key: Optional[Text] = None,
        data: Optional[Data] = None,
    ) -> None:
        """
        Initializes the RasaModelData object.

        Args:
            label_key: the key of a label used for balancing, etc.
            label_sub_key: the sub key of a label used for balancing, etc.
            data: the data holding the features
        """
        self.data = data or defaultdict(lambda: defaultdict(list))
        self.label_key = label_key
        self.label_sub_key = label_sub_key
        # should be updated when features are added
        self.num_examples = self.number_of_examples()

    def get(
        self, key: Text, sub_key: Optional[Text] = None
    ) -> Union[Dict[Text, List[FeatureArray]], List[FeatureArray]]:
        """Get the data under the given keys.

        Args:
            key: The key.
            sub_key: The optional sub key.

        Returns:
            The requested data.
        """
        if sub_key is None and key in self.data:
            return self.data[key]

        if sub_key and key in self.data and sub_key in self.data[key]:
            return self.data[key][sub_key]

        return []

    def items(self) -> ItemsView:
        """Return the items of the data attribute.

        Returns:
            The items of data.
        """
        return self.data.items()

    def values(self) -> Any:
        """Return the values of the data attribute.

        Returns:
            The values of data.
        """
        return self.data.values()

    def keys(self, key: Optional[Text] = None) -> List[Text]:
        """Return the keys of the data attribute.

        Args:
            key: The optional key.

        Returns:
            The keys of the data.
        """
        if key is None:
            return list(self.data.keys())

        if key in self.data:
            return list(self.data[key].keys())

        return []

    def sort(self):
        """Sorts data according to its keys."""
        for key, attribute_data in self.data.items():
            self.data[key] = OrderedDict(sorted(attribute_data.items()))
        self.data = OrderedDict(sorted(self.data.items()))

    def first_data_example(self) -> Data:
        """Return the data with just one feature example per key, sub-key.

        Returns:
            The simplified data.
        """
        out_data = {}
        for key, attribute_data in self.data.items():
            out_data[key] = {}
            for sub_key, features in attribute_data.items():
                out_data[key][sub_key] = [feature[:1] for feature in features]
        return out_data

    def does_feature_exist(self, key: Text, sub_key: Optional[Text] = None) -> bool:
        """Check if feature key (and sub-key) is present and features are available.

        Args:
            key: The key.
            sub_key: The optional sub-key.

        Returns:
            False, if no features for the given keys exists, True otherwise.
        """
        return not self.does_feature_not_exist(key, sub_key)

    def does_feature_not_exist(self, key: Text, sub_key: Optional[Text] = None) -> bool:
        """Check if feature key (and sub-key) is present and features are available.

        Args:
            key: The key.
            sub_key: The optional sub-key.

        Returns:
            True, if no features for the given keys exists, False otherwise.
        """
        if sub_key:
            return (
                key not in self.data
                or not self.data[key]
                or sub_key not in self.data[key]
                or not self.data[key][sub_key]
            )

        return key not in self.data or not self.data[key]

    def is_empty(self) -> bool:
        """Checks if data is set."""

        return not self.data

    def number_of_examples(self, data: Optional[Data] = None) -> int:
        """Obtain number of examples in data.

        Args:
            data: The data.

        Raises: A ValueError if number of examples differ for different features.

        Returns:
            The number of examples in data.
        """
        if not data:
            data = self.data

        if not data:
            return 0

        example_lengths = [
            len(f)
            for attribute_data in data.values()
            for features in attribute_data.values()
            for f in features
        ]

        if not example_lengths:
            return 0

        # check if number of examples is the same for all values
        if not all(length == example_lengths[0] for length in example_lengths):
            raise ValueError(
                f"Number of examples differs for keys '{data.keys()}'. Number of "
                f"examples should be the same for all data."
            )

        return example_lengths[0]

    def number_of_units(self, key: Text, sub_key: Text) -> int:
        """Get the number of units of the given key.

        Args:
            key: The key.
            sub_key: The optional sub-key.

        Returns:
            The number of units.
        """
        if key not in self.data or sub_key not in self.data[key]:
            return 0

        units = 0
        for features in self.data[key][sub_key]:
            if len(features) > 0:
                units += features.units

        return units

    def add_data(self, data: Data, key_prefix: Optional[Text] = None) -> None:
        """Add incoming data to data.

        Args:
            data: The data to add.
            key_prefix: Optional key prefix to use in front of the key value.
        """
        for key, attribute_data in data.items():
            for sub_key, features in attribute_data.items():
                if key_prefix:
                    self.add_features(f"{key_prefix}{key}", sub_key, features)
                else:
                    self.add_features(key, sub_key, features)

    def update_key(
        self, from_key: Text, from_sub_key: Text, to_key: Text, to_sub_key: Text
    ) -> None:
        """Copies the features under the given keys to the new keys and deletes the old keys.

        Args:
            from_key: current feature key
            from_sub_key: current feature sub-key
            to_key: new key for feature
            to_sub_key: new sub-key for feature
        """
        if from_key not in self.data or from_sub_key not in self.data[from_key]:
            return

        if to_key not in self.data:
            self.data[to_key] = {}
        self.data[to_key][to_sub_key] = self.get(from_key, from_sub_key)
        del self.data[from_key][from_sub_key]

        if not self.data[from_key]:
            del self.data[from_key]

    def add_features(
        self, key: Text, sub_key: Text, features: Optional[List[FeatureArray]]
    ) -> None:
        """Add list of features to data under specified key.

        Should update number of examples.

        Args:
            key: The key
            sub_key: The sub-key
            features: The features to add.
        """
        if features is None:
            return

        for feature_array in features:
            if len(feature_array) > 0:
                self.data[key][sub_key].append(feature_array)

        if not self.data[key][sub_key]:
            del self.data[key][sub_key]

        # update number of examples
        self.num_examples = self.number_of_examples()

    def add_lengths(
        self, key: Text, sub_key: Text, from_key: Text, from_sub_key: Text
    ) -> None:
        """Adds a feature array of lengths of sequences to data under given key.

        Args:
            key: The key to add the lengths to
            sub_key: The sub-key to add the lengths to
            from_key: The key to take the lengths from
            from_sub_key: The sub-key to take the lengths from
        """
        if not self.data.get(from_key) or not self.data.get(from_key, {}).get(
            from_sub_key
        ):
            return

        self.data[key][sub_key] = []

        for features in self.data[from_key][from_sub_key]:
            if len(features) == 0:
                continue

            if features.number_of_dimensions == 4:
                lengths = FeatureArray(
                    np.array(
                        [
                            # add one more dim so that dialogue dim
                            # would be a sequence
                            np.array([[[x.shape[0]]] for x in _features])
                            for _features in features
                        ]
                    ),
                    number_of_dimensions=4,
                )
            else:
                lengths = FeatureArray(
                    np.array([x.shape[0] for x in features]), number_of_dimensions=1
                )
            self.data[key][sub_key].extend([lengths])
            break

    def split(
        self, number_of_test_examples: int, random_seed: int
    ) -> Tuple["RasaModelData", "RasaModelData"]:
        """Create random hold out test set using stratified split.

        Args:
            number_of_test_examples: Number of test examples.
            random_seed: Random seed.

        Returns:
            A tuple of train and test RasaModelData.
        """
        self._check_label_key()

        if self.label_key is None or self.label_sub_key is None:
            # randomly split data as no label key is set
            multi_values = [
                v
                for attribute_data in self.data.values()
                for data in attribute_data.values()
                for v in data
            ]
            solo_values = [
                []
                for attribute_data in self.data.values()
                for data in attribute_data.values()
                for _ in data
            ]
            stratify = None
        else:
            # make sure that examples for each label value are in both split sets
            label_ids = self._create_label_ids(
                self.data[self.label_key][self.label_sub_key][0]
            )
            label_counts = dict(zip(*np.unique(label_ids, return_counts=True, axis=0)))

            self._check_train_test_sizes(number_of_test_examples, label_counts)

            counts = np.array([label_counts[label] for label in label_ids])
            # we perform stratified train test split,
            # which insures every label is present in the train and test data
            # this operation can be performed only for labels
            # that contain several data points
            multi_values = [
                f[counts > 1]
                for attribute_data in self.data.values()
                for features in attribute_data.values()
                for f in features
            ]
            # collect data points that are unique for their label
            solo_values = [
                f[counts == 1]
                for attribute_data in self.data.values()
                for features in attribute_data.values()
                for f in features
            ]

            stratify = label_ids[counts > 1]

        output_values = train_test_split(
            *multi_values,
            test_size=number_of_test_examples,
            random_state=random_seed,
            stratify=stratify,
        )

        return self._convert_train_test_split(output_values, solo_values)

    def get_signature(
        self, data: Optional[Data] = None
    ) -> Dict[Text, Dict[Text, List[FeatureSignature]]]:
        """Get signature of RasaModelData.

        Signature stores the shape and whether features are sparse or not for every key.

        Returns:
            A dictionary of key and sub-key to a list of feature signatures
            (same structure as the data attribute).
        """
        if not data:
            data = self.data

        return {
            key: {
                sub_key: [
                    FeatureSignature(f.is_sparse, f.units, f.number_of_dimensions)
                    for f in features
                ]
                for sub_key, features in attribute_data.items()
            }
            for key, attribute_data in data.items()
        }

    def as_tf_dataset(
        self, batch_size: int, batch_strategy: Text = SEQUENCE, shuffle: bool = False
    ) -> tf.data.Dataset:
        """Create tf dataset.

        Args:
            batch_size: The batch size to use.
            batch_strategy: The batch strategy to use.
            shuffle: Boolean indicating whether the data should be shuffled or not.

        Returns:
            The tf.data.Dataset.
        """
        shapes, types = self._get_shapes_types()

        return tf.data.Dataset.from_generator(
            lambda batch_size_: self._gen_batch(batch_size_, batch_strategy, shuffle),
            output_types=types,
            output_shapes=shapes,
            args=([batch_size]),
        )

    def prepare_batch(
        self,
        data: Optional[Data] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        tuple_sizes: Optional[Dict[Text, int]] = None,
    ) -> Tuple[Optional[np.ndarray]]:
        """Slices model data into batch using given start and end value.

        Args:
            data: The data to prepare.
            start: The start index of the batch
            end: The end index of the batch
            tuple_sizes: In case the feature is not present we propagate the batch with
              None. Tuple sizes contains the number of how many None values to add for
              what kind of feature.

        Returns:
            The features of the batch.
        """
        if not data:
            data = self.data

        batch_data = []

        for key, attribute_data in data.items():
            for sub_key, f_data in attribute_data.items():
                # add None for not present values during processing
                if not f_data:
                    if tuple_sizes:
                        batch_data += [None] * tuple_sizes[key]
                    else:
                        batch_data.append(None)
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

                    if _data.is_sparse:
                        batch_data.extend(self._scipy_matrix_to_values(_data))
                    else:
                        batch_data.append(self._pad_dense_data(_data))

        # len of batch_data is equal to the number of keys in model data
        return tuple(batch_data)

    def _get_shapes_types(self) -> Tuple:
        """Extract shapes and types from model data.

        Returns:
            A tuple of shapes and a tuple of types.
        """
        types = []
        shapes = []

        for attribute_data in self.data.values():
            for features in attribute_data.values():
                for f in features:
                    _shapes, _types = f.get_shape_type_info()
                    shapes.extend(_shapes)
                    types.extend(_types)

        return tuple(shapes), tuple(types)

    def _shuffled_data(self, data: Data) -> Data:
        """Shuffle model data.

        Args:
            data: The data to shuffle

        Returns:
            The shuffled data.
        """
        ids = np.random.permutation(self.num_examples)
        return self._data_for_ids(data, ids)

    def _balanced_data(self, data: Data, batch_size: int, shuffle: bool) -> Data:
        """Mix model data to account for class imbalance.

        This batching strategy puts rare classes approximately in every other batch,
        by repeating them. Mimics stratified batching, but also takes into account
        that more populated classes should appear more often.

        Args:
            data: The data.
            batch_size: The batch size.
            shuffle: Boolean indicating whether to shuffle the data or not.

        Returns:
            The balanced data.
        """
        self._check_label_key()

        # skip balancing if labels are token based
        if (
            self.label_key is None
            or self.label_sub_key is None
            or data[self.label_key][self.label_sub_key][0][0].size > 1
        ):
            return data

        label_ids = self._create_label_ids(data[self.label_key][self.label_sub_key][0])

        unique_label_ids, counts_label_ids = np.unique(
            label_ids, return_counts=True, axis=0
        )
        num_label_ids = len(unique_label_ids)

        # group data points by their label
        # need to call every time, so that the data is shuffled inside each class
        data_by_label = self._split_by_label_ids(data, label_ids, unique_label_ids)

        # running index inside each data grouped by labels
        data_idx = [0] * num_label_ids
        # number of cycles each label was passed
        num_data_cycles = [0] * num_label_ids
        # if a label was skipped in current batch
        skipped = [False] * num_label_ids

        new_data = defaultdict(lambda: defaultdict(list))

        while min(num_data_cycles) == 0:
            if shuffle:
                indices_of_labels = np.random.permutation(num_label_ids)
            else:
                indices_of_labels = range(num_label_ids)

            for index in indices_of_labels:
                if num_data_cycles[index] > 0 and not skipped[index]:
                    skipped[index] = True
                    continue

                skipped[index] = False

                index_batch_size = (
                    int(counts_label_ids[index] / self.num_examples * batch_size) + 1
                )

                for key, attribute_data in data_by_label[index].items():
                    for sub_key, features in attribute_data.items():
                        for i, f in enumerate(features):
                            if len(new_data[key][sub_key]) < i + 1:
                                new_data[key][sub_key].append([])
                            new_data[key][sub_key][i].append(
                                f[data_idx[index] : data_idx[index] + index_batch_size]
                            )

                data_idx[index] += index_batch_size
                if data_idx[index] >= counts_label_ids[index]:
                    num_data_cycles[index] += 1
                    data_idx[index] = 0

                if min(num_data_cycles) > 0:
                    break

        final_data = defaultdict(lambda: defaultdict(list))
        for key, attribute_data in new_data.items():
            for sub_key, features in attribute_data.items():
                for f in features:
                    final_data[key][sub_key].append(
                        FeatureArray(
                            np.concatenate(np.array(f)),
                            number_of_dimensions=f[0].number_of_dimensions,
                        )
                    )

        return final_data

    def _gen_batch(
        self, batch_size: int, batch_strategy: Text = SEQUENCE, shuffle: bool = False
    ) -> Generator[Tuple[Optional[np.ndarray]], None, None]:
        """Generate batches.

        Args:
            batch_size: The batch size
            batch_strategy: The batch strategy.
            shuffle: Boolean indicating whether to shuffle the data or not.

        Returns:
            A generator over the batches.
        """
        data = self.data
        num_examples = self.num_examples

        if shuffle:
            data = self._shuffled_data(data)

        if batch_strategy == BALANCED:
            data = self._balanced_data(data, batch_size, shuffle)
            # after balancing, number of examples increased
            num_examples = self.number_of_examples(data)

        num_batches = num_examples // batch_size + int(num_examples % batch_size > 0)

        for batch_num in range(num_batches):
            start = batch_num * batch_size
            end = start + batch_size

            yield self.prepare_batch(data, start, end)

    def _check_train_test_sizes(
        self, number_of_test_examples: int, label_counts: Dict[Any, int]
    ) -> None:
        """Check whether the test data set is too large or too small.

        Args:
            number_of_test_examples: number of test examples
            label_counts: number of labels

        Raises:
            A ValueError if the number of examples does not fit.
        """
        if number_of_test_examples >= self.num_examples - len(label_counts):
            raise ValueError(
                f"Test set of {number_of_test_examples} is too large. Remaining "
                f"train set should be at least equal to number of classes "
                f"{len(label_counts)}."
            )
        if number_of_test_examples < len(label_counts):
            raise ValueError(
                f"Test set of {number_of_test_examples} is too small. It should "
                f"be at least equal to number of classes {label_counts}."
            )

    @staticmethod
    def _data_for_ids(data: Optional[Data], ids: np.ndarray) -> Data:
        """Filter model data by ids.

        Args:
            data: The data to filter
            ids: The ids

        Returns:
            The filtered data
        """
        new_data = defaultdict(lambda: defaultdict(list))

        if data is None:
            return new_data

        for key, attribute_data in data.items():
            for sub_key, features in attribute_data.items():
                for f in features:
                    new_data[key][sub_key].append(f[ids])
        return new_data

    def _split_by_label_ids(
        self, data: Optional[Data], label_ids: np.ndarray, unique_label_ids: np.ndarray
    ) -> List["RasaModelData"]:
        """Reorganize model data into a list of model data with the same labels.

        Args:
            data: The data
            label_ids: The label ids
            unique_label_ids: The unique label ids

        Returns:
            Reorganized RasaModelData
        """
        label_data = []
        for label_id in unique_label_ids:
            matching_ids = np.array(label_ids) == label_id
            label_data.append(
                RasaModelData(
                    self.label_key,
                    self.label_sub_key,
                    self._data_for_ids(data, matching_ids),
                )
            )
        return label_data

    def _check_label_key(self) -> None:
        """Check if the label key exists.

        Raises:
            ValueError if the label key and sub-key is not in data.
        """
        if (
            self.label_key is not None
            and self.label_sub_key is not None
            and (
                self.label_key not in self.data
                or self.label_sub_key not in self.data[self.label_key]
                or len(self.data[self.label_key][self.label_sub_key]) > 1
            )
        ):
            raise ValueError(
                f"Key '{self.label_key}.{self.label_sub_key}' not in RasaModelData."
            )

    def _convert_train_test_split(
        self, output_values: List[Any], solo_values: List[Any]
    ) -> Tuple["RasaModelData", "RasaModelData"]:
        """Converts the output of sklearn's train_test_split into model data.

        Args:
            output_values: output values of sklearn's train_test_split
            solo_values: list of solo values

        Returns:
            The test and train RasaModelData
        """
        data_train = defaultdict(lambda: defaultdict(list))
        data_val = defaultdict(lambda: defaultdict(list))

        # output_values = x_train, x_val, y_train, y_val, z_train, z_val, etc.
        # order is kept, e.g. same order as model data keys

        # train datasets have an even index
        index = 0
        for key, attribute_data in self.data.items():
            for sub_key, features in attribute_data.items():
                for f in features:
                    data_train[key][sub_key].append(
                        self._combine_features(
                            output_values[index * 2],
                            solo_values[index],
                            f.number_of_dimensions,
                        )
                    )
                    index += 1

        # val datasets have an odd index
        index = 0
        for key, attribute_data in self.data.items():
            for sub_key, features in attribute_data.items():
                for _ in features:
                    data_val[key][sub_key].append(output_values[(index * 2) + 1])
                    index += 1

        return (
            RasaModelData(self.label_key, self.label_sub_key, data_train),
            RasaModelData(self.label_key, self.label_sub_key, data_val),
        )

    @staticmethod
    def _combine_features(
        feature_1: Union[np.ndarray, scipy.sparse.spmatrix],
        feature_2: Union[np.ndarray, scipy.sparse.spmatrix],
        number_of_dimensions: Optional[int] = 1,
    ) -> FeatureArray:
        """Concatenate features.

        Args:
            feature_1: Features to concatenate.
            feature_2: Features to concatenate.

        Returns:
            The combined features.
        """
        if isinstance(feature_1, scipy.sparse.spmatrix) and isinstance(
            feature_2, scipy.sparse.spmatrix
        ):
            if feature_2.shape[0] == 0:
                return FeatureArray(feature_1, number_of_dimensions)
            if feature_1.shape[0] == 0:
                return FeatureArray(feature_2, number_of_dimensions)
            return FeatureArray(
                scipy.sparse.vstack([feature_1, feature_2]), number_of_dimensions
            )

        return FeatureArray(
            np.concatenate([feature_1, feature_2]), number_of_dimensions
        )

    @staticmethod
    def _create_label_ids(label_ids: FeatureArray) -> np.ndarray:
        """Convert various size label_ids into single dim array.

        For multi-label y, map each distinct row to a string representation
        using join because str(row) uses an ellipsis if len(row) > 1000.
        Idea taken from sklearn's stratify split.

        Args:
            label_ids: The label ids.

        Raises:
            ValueError if dimensionality of label ids is not supported

        Returns:
            The single dim label array.
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
            return RasaModelData._pad_4d_dense_data(array_of_dense)

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
    def _pad_4d_dense_data(array_of_array_of_dense: FeatureArray) -> np.ndarray:
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
        number_of_features = array_of_array_of_dense[0][0].shape[-1]
        array_of_array_of_dense = RasaModelData._filter_out_fake_inputs(
            array_of_array_of_dense
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
            return RasaModelData._4d_scipy_matrix_to_values(array_of_sparse)

        # we need to make sure that the matrices are coo_matrices otherwise the
        # transformation does not work (e.g. you cannot access x.row, x.col)
        if not isinstance(array_of_sparse[0], scipy.sparse.coo_matrix):
            array_of_sparse = [x.tocoo() for x in array_of_sparse]

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
    def _4d_scipy_matrix_to_values(
        array_of_array_of_sparse: FeatureArray,
    ) -> List[np.ndarray]:
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
        number_of_features = array_of_array_of_sparse[0][0].shape[-1]
        array_of_array_of_sparse = RasaModelData._filter_out_fake_inputs(
            array_of_array_of_sparse
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
                [x.tocoo() for x in array_of_sparse]
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
