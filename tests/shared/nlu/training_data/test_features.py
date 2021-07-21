import subprocess
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

from rasa.shared.nlu.training_data.features import Features


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

    features = [
        Features(m1, FEATURE_TYPE_SENTENCE, TEXT, "CountVectorsFeaturizer"),
        Features(m2, FEATURE_TYPE_SENTENCE, TEXT, "CountVectorsFeaturizer"),
        Features(m1, FEATURE_TYPE_SEQUENCE, TEXT, "CountVectorsFeaturizer"),
        Features(m1, FEATURE_TYPE_SEQUENCE, TEXT, "RegexFeaturizer"),
        Features(m1, FEATURE_TYPE_SENTENCE, INTENT, "CountVectorsFeaturizer"),
        Features(
            scipy.sparse.coo_matrix(m1),
            FEATURE_TYPE_SENTENCE,
            INTENT,
            "CountVectorsFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m2),
            FEATURE_TYPE_SENTENCE,
            INTENT,
            "CountVectorsFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m1),
            FEATURE_TYPE_SENTENCE,
            TEXT,
            "CountVectorsFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m1), FEATURE_TYPE_SENTENCE, TEXT, "RegexFeaturizer",
        ),
        Features(
            scipy.sparse.coo_matrix(m1),
            FEATURE_TYPE_SEQUENCE,
            TEXT,
            "CountVectorsFeaturizer",
        ),
    ]
    fingerprints = {f.fingerprint() for f in features}
    assert len(fingerprints) == len(features)


def test_dense_feature_fingerprints_are_consistent_across_runs():
    """Tests that fingerprints are consistent across python interpreter invocations."""
    # unfortunately, monkeypatching PYTHONHASHSEED does not work in a running process
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    cmd = """python -c \
        'import numpy as np; \
        from rasa.shared.nlu.training_data.features import Features; \
        m = np.asarray([[0.5, 3.1, 3.0], [1.1, 1.2, 1.3], [4.7, 0.3, 2.7]]); \
        feature = Features(m, "sentence", "text", "CountVectorsFeaturizer"); \
        print(feature.fingerprint())'"""

    fp1 = subprocess.getoutput(cmd)
    fp2 = subprocess.getoutput(cmd)
    print(fp1)
    print(fp2)
    assert len(fp1) == 32
    assert fp1 == fp2


def test_sparse_feature_fingerprints_are_consistent_across_runs():
    """Tests that fingerprints are consistent across python interpreter invocations."""
    # unfortunately, monkeypatching PYTHONHASHSEED does not work in a running process
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    cmd = """python -c \
        'import numpy as np; \
        import scipy.sparse; \
        from rasa.shared.nlu.training_data.features import Features; \
        m = np.asarray([[0.5, 3.1, 3.0], [1.1, 1.2, 1.3], [4.7, 0.3, 2.7]]); \
        m_sparse = scipy.sparse.coo_matrix(m); \
        feature = Features(m_sparse, "sentence", "text", "CountVectorsFeaturizer"); \
        print(feature.fingerprint())'"""

    fp1 = subprocess.getoutput(cmd)
    fp2 = subprocess.getoutput(cmd)
    print(fp1)
    print(fp2)
    assert len(fp1) == 32
    assert fp1 == fp2


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
