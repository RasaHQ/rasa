from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import numpy as np
import pytest

from rasa_core.domain import TemplateDomain
from rasa_core.featurizers import BinaryFeaturizer
from rasa_core.policies.ensemble import PolicyEnsemble
from rasa_core.slots import Slot
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
    domain_file = "examples/restaurant_domain.yml"
    domain = TemplateDomain.load(domain_file)
    assert len(domain.intents) == 6
    assert len(domain.actions) == 18


def test_utter_templates():
    domain_file = "examples/restaurant_domain.yml"
    domain = TemplateDomain.load(domain_file)
    expected_template = {
        "text": "in which price range?",
        "buttons": [{"title": "cheap", "payload": "cheap"},
                    {"title": "expensive", "payload": "expensive"}]
    }
    assert domain.random_template_for("utter_ask_price") == expected_template


def test_restaurant_domain_is_valid():
    # should raise no exception
    TemplateDomain.validate_domain_yaml('examples/restaurant_domain.yml')


def write_domain_yml(tmpdir, yml):
    path = tmpdir.join("domain.yml").strpath
    with io.open(path, "w") as f:
        f.write(yml)
    return path


def test_custom_slot_type(tmpdir):
    domain_path = write_domain_yml(tmpdir, """
       slots:
         custom:
           type: tests.conftest.CustomSlot

       templates:
         utter_greet:
           - hey there!

       actions:
         - utter_greet """)
    TemplateDomain.load(domain_path)


def test_domain_fails_on_unknown_custom_slot_type(tmpdir):
    domain_path=write_domain_yml(tmpdir,"""
        slots:
            custom:
             type: tests.conftest.Unknown
        
        templates:
            utter_greet:
             - hey there!
        
        actions:
            - utter_greet""")
    with pytest.raises(ValueError):
        TemplateDomain.load(domain_path)
