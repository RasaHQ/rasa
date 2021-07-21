import subprocess

import numpy as np
import scipy.sparse
from rasa.shared.nlu.constants import (
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    TEXT,
    INTENT,
)

from rasa.shared.nlu.training_data.features import Features


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


def test_feature_fingerprints_are_consistent_across_runs():
    """Tests that fingerprints are consistent across python interpreter invocations."""
    # unfortunately, monkeypatching PYTHONHASHSEED does not work in a running process
    # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    cmd = """python -c \
        'import numpy as np; \
        from rasa.shared.nlu.training_data.features import Features; \
        m1 = np.asarray([[0.5, 3.1, 3.0], [1.1, 1.2, 1.3], [4.7, 0.3, 2.7]]); \
        feature = Features(m1, "sentence", "text", "CountVectorsFeaturizer"); \
        print(feature.fingerprint())'"""

    fp1 = subprocess.getoutput(cmd)
    fp2 = subprocess.getoutput(cmd)
    print(fp1)
    print(fp2)
    assert len(fp1) == 32
    assert fp1 == fp2


def test_feature_fingerprints_take_into_account_full_array():
    """Tests that fingerprint isn't using summary/abbreviated array info."""
    big_array = np.random.random((128, 128, 8))

    f1 = Features(big_array, FEATURE_TYPE_SENTENCE, TEXT, "RegexFeaturizer")
    big_array_with_zero = np.copy(big_array)
    big_array_with_zero[64, 64, 4] = 0.0
    f2 = Features(big_array_with_zero, FEATURE_TYPE_SENTENCE, TEXT, "RegexFeaturizer")

    assert f1.fingerprint() != f2.fingerprint()
