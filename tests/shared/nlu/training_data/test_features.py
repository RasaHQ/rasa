import copy
import itertools
from typing import Optional, Text, List, Dict, Tuple, Any, Callable

import numpy as np
import pytest
import scipy.sparse

from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    TEXT,
    INTENT,
)


@pytest.mark.parametrize(
    "type,is_sparse,",
    itertools.product([FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE], [True, False]),
)
def test_print(type: Text, is_sparse: bool):
    first_dim = 1 if type == FEATURE_TYPE_SEQUENCE else 3
    matrix = np.full(shape=(first_dim, 2), fill_value=1)
    if is_sparse:
        matrix = scipy.sparse.coo_matrix(matrix)
    feat = Features(
        features=matrix,
        attribute="fixed-attribute",
        feature_type=type,
        origin="origin--doesn't-matter-here",
    )
    assert repr(feat)
    assert str(feat)


def test_combine_with_existing_dense_features():
    existing_features = Features(
        np.array([[1, 0, 2, 3], [2, 0, 0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "test"
    )
    new_features = Features(
        np.array([[1, 0], [0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin"
    )
    expected_features = np.array([[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]])

    existing_features.combine_with_features(new_features)

    assert np.all(expected_features == existing_features.features)


def test_combine_with_existing_dense_features_shape_mismatch():
    existing_features = Features(
        np.array([[1, 0, 2, 3], [2, 0, 0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "test"
    )
    new_features = Features(np.array([[0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin")

    with pytest.raises(ValueError):
        existing_features.combine_with_features(new_features)


def test_combine_with_existing_sparse_features():
    existing_features = Features(
        scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
        FEATURE_TYPE_SEQUENCE,
        TEXT,
        "test",
    )
    new_features = Features(
        scipy.sparse.csr_matrix([[1, 0], [0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin"
    )
    expected_features = [[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]]

    existing_features.combine_with_features(new_features)
    actual_features = existing_features.features.toarray()

    assert np.all(expected_features == actual_features)


def test_combine_with_existing_sparse_features_shape_mismatch():
    existing_features = Features(
        scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
        FEATURE_TYPE_SEQUENCE,
        TEXT,
        "test",
    )
    new_features = Features(
        scipy.sparse.csr_matrix([[0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin"
    )

    with pytest.raises(ValueError):
        existing_features.combine_with_features(new_features)


def test_for_features_fingerprinting_collisions():
    """Tests that features fingerprints are unique."""
    m1 = np.asarray([[0.5, 3.1, 3.0], [1.1, 1.2, 1.3], [4.7, 0.3, 2.7]])

    m2 = np.asarray([[0, 0, 0], [1, 2, 3], [0, 0, 1]])

    dense_features = [
        Features(m1, FEATURE_TYPE_SENTENCE, TEXT, "CountVectorsFeaturizer"),
        Features(m2, FEATURE_TYPE_SENTENCE, TEXT, "CountVectorsFeaturizer"),
        Features(m1, FEATURE_TYPE_SEQUENCE, TEXT, "CountVectorsFeaturizer"),
        Features(m1, FEATURE_TYPE_SEQUENCE, TEXT, "RegexFeaturizer"),
        Features(m1, FEATURE_TYPE_SENTENCE, INTENT, "CountVectorsFeaturizer"),
    ]
    dense_fingerprints = {f.fingerprint() for f in dense_features}
    assert len(dense_fingerprints) == len(dense_features)

    sparse_features = [
        Features(
            scipy.sparse.coo_matrix(m1),
            FEATURE_TYPE_SENTENCE,
            TEXT,
            "CountVectorsFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m2),
            FEATURE_TYPE_SENTENCE,
            TEXT,
            "CountVectorsFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m1),
            FEATURE_TYPE_SEQUENCE,
            TEXT,
            "CountVectorsFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m1), FEATURE_TYPE_SEQUENCE, TEXT, "RegexFeaturizer"
        ),
        Features(
            scipy.sparse.coo_matrix(m1),
            FEATURE_TYPE_SENTENCE,
            INTENT,
            "CountVectorsFeaturizer",
        ),
    ]
    sparse_fingerprints = {f.fingerprint() for f in sparse_features}
    assert len(sparse_fingerprints) == len(sparse_features)


def test_feature_fingerprints_take_into_account_full_array():
    """Tests that fingerprint isn't using summary/abbreviated array info."""
    big_array = np.random.random((128, 128))

    f1 = Features(big_array, FEATURE_TYPE_SENTENCE, TEXT, "RegexFeaturizer")
    big_array_with_zero = np.copy(big_array)
    big_array_with_zero[64, 64] = 0.0
    f2 = Features(big_array_with_zero, FEATURE_TYPE_SENTENCE, TEXT, "RegexFeaturizer")

    assert f1.fingerprint() != f2.fingerprint()

    f1_sparse = Features(
        scipy.sparse.coo_matrix(big_array),
        FEATURE_TYPE_SENTENCE,
        TEXT,
        "RegexFeaturizer",
    )

    f2_sparse = Features(
        scipy.sparse.coo_matrix(big_array_with_zero),
        FEATURE_TYPE_SENTENCE,
        TEXT,
        "RegexFeaturizer",
    )

    assert f1_sparse.fingerprint() != f2_sparse.fingerprint()


def _consistent_features_list(
    is_sparse: bool, feature_type: Text, length: int
) -> List[Features]:
    """Creates a list of features with the required properties.

    Args:
        is_sparse: whether all features should be sparse
        type: the type to be used for all features
        length: the number of features to generate
    Returns:
      a tuple containing a list of features with the requested attributes
    """
    first_dim = 1 if feature_type == FEATURE_TYPE_SENTENCE else 3
    # create list of features whose properties match - except the shapes and
    # feature values which are chosen in a specific way
    features_list = []
    for idx in range(length):
        matrix = np.full(shape=(first_dim, idx + 1), fill_value=idx + 1)
        if is_sparse:
            matrix = scipy.sparse.coo_matrix(matrix)
        config = dict(
            features=matrix,
            attribute="fixed-attribute",
            feature_type=feature_type,
            origin=f"origin-{idx}",
        )
        feat = Features(**config)
        features_list.append(feat)
    return features_list


def _change_feature_type(features: Features) -> None:
    other_type = (
        FEATURE_TYPE_SENTENCE
        if features.type == FEATURE_TYPE_SEQUENCE
        else FEATURE_TYPE_SEQUENCE
    )
    other_seq_len = 1 if other_type == FEATURE_TYPE_SENTENCE else 3
    same_dim = features.features.shape[-1]
    other_matrix = np.full(shape=(other_seq_len, same_dim), fill_value=same_dim)
    if features.is_sparse():
        other_matrix = scipy.sparse.coo_matrix(other_matrix)

    features.features = other_matrix
    features.type = other_type


def _change_sparseness(features: Features) -> None:
    if features.is_dense():
        other_matrix = scipy.sparse.coo_matrix(features.features)
    else:
        other_matrix = features.features.todense()
    features.features = other_matrix


def _change_sequence_length(features: Features) -> None:
    if features.type != FEATURE_TYPE_SEQUENCE:
        return
    other_seq_len = features.features.shape[0] + 1
    same_dim = features.features.shape[-1]
    matrix_with_other_seq_len = np.full(
        shape=(other_seq_len, same_dim), fill_value=same_dim
    )
    if features.is_sparse():
        matrix_with_other_seq_len = scipy.sparse.coo_matrix(matrix_with_other_seq_len)
    features.features = matrix_with_other_seq_len


def _change_attribute(features: Features) -> None:
    features.attribute = "OTHER-ATTRIBUTE"


def _change_origin(features: Features) -> None:
    features.origin = "OTHER-ORIGIN"


@pytest.mark.parametrize(
    "is_sparse,feature_type,length,use_expected_origin",
    itertools.product(
        [True, False],
        [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE],
        [1, 2, 5],
        [True, False],
    ),
)
def test_combine(
    is_sparse: bool, feature_type: Text, length: int, use_expected_origin: bool
):

    features_list = _consistent_features_list(
        is_sparse=is_sparse, feature_type=feature_type, length=length
    )
    first_dim = features_list[0].features.shape[0]

    origins = [f"origin-{idx}" for idx in range(len(features_list))]
    expected_origin = origins if use_expected_origin else None

    # works as expected
    if use_expected_origin:
        combination = Features.combine(features_list, expected_origins=expected_origin)
        assert combination.features.shape[1] == int(length * (length + 1) / 2)
        assert combination.features.shape[0] == first_dim
        assert combination.origin == origins
        assert combination.is_sparse() == is_sparse
        matrix = combination.features
        if is_sparse:
            matrix = combination.features.todense()
        for idx in range(length):
            offset = int(idx * (idx + 1) / 2)
            assert np.all(matrix[:, offset : (offset + idx + 1)] == idx + 1)
    else:
        with pytest.raises(ValueError):
            Features.combine(features_list, expected_origins=["unexpected"])


@pytest.mark.parametrize(
    "is_sparse,feature_type,length,modification",
    itertools.product(
        [True, False],
        [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE],
        [1, 2, 5],
        [
            _change_attribute,
            _change_feature_type,
            _change_sparseness,
            _change_sequence_length,
            None,
        ],  # origin is not checked
    ),
)
def test_assert_consistency(
    is_sparse: bool,
    feature_type: Text,
    length: int,
    modification: Optional[Callable[[Features], None]],
):
    features_list = _consistent_features_list(
        is_sparse=is_sparse, feature_type=feature_type, length=length
    )
    if (
        modification is not None
        and length > 1
        and not (
            feature_type == FEATURE_TYPE_SENTENCE
            and modification == _change_sequence_length
        )
    ):
        modification(features_list[-1])
        with pytest.raises(ValueError):
            Features.assert_consistency(features_list)
    else:
        Features.assert_consistency(features_list)


@pytest.mark.parametrize(
    "is_sparse,feature_type,length",
    itertools.product(
        [True, False],
        [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE],
        [1, 2, 5],
    ),
)
def test_filter_passes_all(is_sparse: bool, feature_type: Text, length: int):

    features_list = _consistent_features_list(
        is_sparse=is_sparse, feature_type=feature_type, length=length
    )
    result = Features.filter(
        features_list,
        attributes=["fixed-attribute"],
        feature_type=feature_type,
        is_sparse=is_sparse,
    )
    assert len(result) == length


@pytest.mark.parametrize(
    "is_sparse,feature_type,length,modification",
    itertools.product(
        [True, False],
        [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE],
        [3],
        [_change_feature_type, _change_attribute, _change_sparseness],
    ),
)
def test_filter_removes_some(
    is_sparse: bool,
    feature_type: Text,
    length: int,
    modification: Callable[[Features], None],
):
    features_list = _consistent_features_list(
        is_sparse=is_sparse, feature_type=feature_type, length=length
    )
    modification(features_list[-1])

    result = Features.filter(
        features_list,
        attributes=["fixed-attribute"],
        feature_type=feature_type,
        is_sparse=is_sparse,
    )
    assert len(result) == len(features_list) - 1

    # don't forget to check the origin
    filter_config = dict(
        attributes=["fixed-attribute"],
        feature_type=feature_type,
        origin=["origin-0"],
        is_sparse=is_sparse,
    )
    result = Features.filter(features_list, **filter_config)
    assert len(result) == 1


@pytest.mark.parametrize(
    "num_features_per_attribute,specified_attributes",
    itertools.product(
        [{"a": 3, "b": 1, "c": 0}],
        [None, ["a", "b", "c", "doesnt-appear"], ["doesnt-appear"]],
    ),
)
def test_groupby(
    num_features_per_attribute: Dict[Text, int],
    specified_attributes: Optional[List[Text]],
):

    features_list = []
    for attribute, number in num_features_per_attribute.items():
        for idx in range(number):
            matrix = np.full(shape=(1, idx + 1), fill_value=idx + 1)
            config = dict(
                features=matrix,
                attribute=attribute,
                feature_type=FEATURE_TYPE_SEQUENCE,  # doesn't matter
                origin=f"origin-{idx}",  # doens't matter
            )
            feat = Features(**config)
            features_list.append(feat)

    result = Features.groupby_attribute(features_list, attributes=specified_attributes)
    if specified_attributes is None:
        for attribute, number in num_features_per_attribute.items():
            if number > 0:
                assert attribute in result
                assert len(result[attribute]) == number
            else:
                assert attribute not in result
    else:
        assert set(result.keys()) == set(specified_attributes)
        for attribute in specified_attributes:
            assert attribute in result
            number = num_features_per_attribute.get(attribute, 0)
            assert len(result[attribute]) == number


@pytest.mark.parametrize(
    "shuffle_mode,num_features_per_combination",
    itertools.product(
        ["reversed", "random"], [[1, 0, 0, 0], [1, 1, 1, 1], [2, 3, 4, 5], [0, 1, 2, 2]]
    ),
)
def test_reduce(
    shuffle_mode: Text, num_features_per_combination: Tuple[int, int, int, int]
):

    # all combinations - in the expected order
    # (i.e. all sparse before all dense and sequence before sentence)
    all_combinations = [
        (FEATURE_TYPE_SEQUENCE, True),
        (FEATURE_TYPE_SENTENCE, True),
        (FEATURE_TYPE_SEQUENCE, False),
        (FEATURE_TYPE_SENTENCE, False),
    ]

    # multiply accordingly and mess up the order
    chosen_combinations = [
        spec
        for spec, num in zip(all_combinations, num_features_per_combination)
        for _ in range(num)
    ]
    if shuffle_mode == "reversed":
        messed_up_order = reversed(chosen_combinations)
    else:
        # Note: rng.permutation would mess up the types
        rng = np.random.default_rng(23452345)
        permutation = rng.permutation(len(chosen_combinations))
        messed_up_order = [chosen_combinations[idx] for idx in permutation]

    # create features accordingly
    features_list = []
    for idx, (type, is_sparse) in enumerate(messed_up_order):
        first_dim = 1 if type == FEATURE_TYPE_SEQUENCE else 3
        matrix = np.full(shape=(first_dim, 1), fill_value=1)
        if is_sparse:
            matrix = scipy.sparse.coo_matrix(matrix)
        config = dict(
            features=matrix,
            attribute="fixed-attribute",  # must be the same
            feature_type=type,
            origin="origin-does-matter-here",  # must be the same
        )
        feat = Features(**config)
        features_list.append(feat)

    # reduce!
    reduced_list = Features.reduce(features_list)
    assert len(reduced_list) == sum(num > 0 for num in num_features_per_combination)
    idx = 0
    for num, (type, is_sparse) in zip(num_features_per_combination, all_combinations):
        if num == 0:
            # nothing to check here - because we already checked the length above
            # and check the types and shape of all existing features in this loop
            pass
        else:
            feature = reduced_list[idx]
            assert feature.is_sparse() == is_sparse
            assert feature.type == type
            assert feature.features.shape[-1] == num
            idx += 1


@pytest.mark.parametrize("differ", ["attribute", "origin"])
def test_reduce_raises_if_combining_different_origins_or_attributes(differ: Text):
    # create features accordingly
    arbitrary_fixed_type = FEATURE_TYPE_SENTENCE
    features_list = []
    for idx in range(2):
        first_dim = 1
        arbitrary_matrix_matching_type = np.full(shape=(first_dim, 1), fill_value=1)
        config = dict(
            features=arbitrary_matrix_matching_type,
            attribute="fixed-attribute" if differ != "attribute" else f"attr-{idx}",
            feature_type=arbitrary_fixed_type,
            origin="fixed-origin" if differ != "origin" else f"origin-{idx}",
        )
        feat = Features(**config)
        features_list.append(feat)

    # reduce!
    if differ == "attribute":
        message = "Expected all Features to describe the same attribute"
        expected_origin = ["origin"]
    else:
        message = "Expected 'origin-1' to be the origin of the 0-th"
        expected_origin = ["origin-1"]
    with pytest.raises(ValueError, match=message):
        Features.reduce(features_list, expected_origins=expected_origin)
