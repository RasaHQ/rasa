from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from rasa_core.domain import TemplateDomain
from rasa_core.featurizers import BinaryFeaturizer
from rasa_core.training_utils import extract_training_data_from_file


def test_create_train_data_no_history(default_domain):
    featurizer = BinaryFeaturizer()
    X, y = extract_training_data_from_file(
            "data/dsl_stories/stories_defaultdomain.md",
            augmentation_factor=0,
            domain=default_domain,
            featurizer=featurizer,
            max_history=1
    )
    reference = np.array([
        [[0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0, 0, 0, 0, 1]],
        [[0, 0, 1, 0, 1, 0, 0, 0, 0]],
        [[0, 1, 0, 0, 0, 0, 1, 0, 0]],
        [[0, 1, 0, 0, 1, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 1, 0]],
        [[1, 0, 0, 0, 1, 0, 0, 0, 0]],
    ])
    assert X.shape == reference.shape
    assert np.array_equal(X, reference)


def test_create_train_data_with_history(default_domain):
    featurizer = BinaryFeaturizer()
    X, y = extract_training_data_from_file(
        "data/dsl_stories/stories_defaultdomain.md",
        augmentation_factor=0,
        domain=default_domain,
        featurizer=featurizer,
        max_history=4
        )
    reference = np.array([
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 1, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 1]],

        [[1, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 0]],

        [[1, 0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 1, 0, 0]],

        [[-1, -1, -1, -1, -1, -1, -1, -1, -1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 1, 0]],

        [[-1, -1, -1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1, -1, -1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]],

        [[-1, -1, -1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1, -1, -1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 0]],
    ])
    assert X.shape == reference.shape
    assert np.array_equal(X, reference)


def test_domain_from_template():
    file = "examples/restaurant_domain.yml"
    domain = TemplateDomain.load(file)
    assert len(domain.intents) == 6
    assert len(domain.actions) == 18
