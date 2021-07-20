from typing import Text
import pytest

import numpy as np
import scipy.sparse
from rasa.shared.nlu.constants import (
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    TEXT,
    INTENT,
)

from rasa.shared.nlu.training_data.features import Features

m1 = [[0.5, 3.1, 3.0], [1.1, 1.2, 1.3], [4.7, 0.3, 2.7]]

m2 = [[0, 0, 0], [1, 2, 3], [0, 0, 1]]


@pytest.mark.parametrize(
    "features, fingerprint",
    [
        (
            Features(m1, FEATURE_TYPE_SENTENCE, TEXT, "CountVectorsFeaturizer"),
            "81419febe54d0011665f983ff9821b49",
        ),
        (
            Features(m2, FEATURE_TYPE_SENTENCE, TEXT, "CountVectorsFeaturizer"),
            "8c377a95390cebc98d8be4ef65681cef",
        ),
        (
            Features(m1, FEATURE_TYPE_SEQUENCE, TEXT, "CountVectorsFeaturizer"),
            "afac05ed1ac11438fdbd3d534ff25c5f",
        ),
        (
            Features(m1, FEATURE_TYPE_SEQUENCE, TEXT, "RegexFeaturizer"),
            "e49727933d89d58c00bbc5829b05e649",
        ),
        (
            Features(m1, FEATURE_TYPE_SENTENCE, INTENT, "CountVectorsFeaturizer",),
            "8046c85b52fc42986178f49fe0570bba",
        ),
        (
            Features(
                scipy.sparse.coo_matrix(m1),
                FEATURE_TYPE_SENTENCE,
                INTENT,
                "CountVectorsFeaturizer",
            ),
            "c1d9087fe74644d162945c15c8864cdb",
        ),
        (
            Features(
                scipy.sparse.coo_matrix(m2),
                FEATURE_TYPE_SENTENCE,
                INTENT,
                "CountVectorsFeaturizer",
            ),
            "37c60e5c96639a1f0719a44d1deb4fc9",
        ),
        (
            Features(
                scipy.sparse.coo_matrix(m1),
                FEATURE_TYPE_SENTENCE,
                TEXT,
                "CountVectorsFeaturizer",
            ),
            "e43a8a0daf4c9df83110d0b4e0bbf366",
        ),
        (
            Features(
                scipy.sparse.coo_matrix(m1),
                FEATURE_TYPE_SENTENCE,
                TEXT,
                "RegexFeaturizer",
            ),
            "d01c782517d5ae61c29a00a88cc305b7",
        ),
        (
            Features(
                scipy.sparse.coo_matrix(m1),
                FEATURE_TYPE_SEQUENCE,
                TEXT,
                "CountVectorsFeaturizer",
            ),
            "9d354bee9a42f74253ce5c47cef621e2",
        ),
    ],
)
def test_features_fingerprinting_consistency(features: Features, fingerprint: Text):
    """Tests that features fingerprints are consistent across runs and machines."""
    assert features.fingerprint() == fingerprint
