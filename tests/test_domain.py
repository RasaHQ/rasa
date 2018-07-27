from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import pytest

from rasa_core import training
from rasa_core.domain import TemplateDomain
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer
from rasa_core.utils import read_file
from tests import utilities
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


def test_create_train_data_no_history(default_domain):
    featurizer = MaxHistoryTrackerFeaturizer(max_history=1)
    training_trackers = training.load_data(
            DEFAULT_STORIES_FILE,
            default_domain,
            augmentation_factor=0
    )
    assert len(training_trackers) == 3
    (decoded, _) = featurizer.training_states_and_actions(
            training_trackers, default_domain)

    # decoded needs to be sorted
    hashed = []
    for states in decoded:
        hashed.append(json.dumps(states, sort_keys=True))
    hashed = sorted(hashed, reverse=True)

    assert hashed == [
        '[{}]',
        '[{"intent_greet": 1.0, "prev_utter_greet": 1.0}]',
        '[{"intent_greet": 1.0, "prev_action_listen": 1.0}]',
        '[{"intent_goodbye": 1.0, "prev_utter_goodbye": 1.0}]',
        '[{"intent_goodbye": 1.0, "prev_action_listen": 1.0}]',
        '[{"intent_default": 1.0, "prev_utter_default": 1.0}]',
        '[{"intent_default": 1.0, "prev_utter_default": 1.0, '
        '"slot_name_0": 1.0}]',
        '[{"intent_default": 1.0, "prev_action_listen": 1.0}]',
        '[{"intent_default": 1.0, "prev_action_listen": 1.0, '
        '"slot_name_0": 1.0}]',
        '[{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_utter_greet": 1.0, "slot_name_0": 1.0}]',
        '[{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}]'
    ]


def test_create_train_data_with_history(default_domain):
    featurizer = MaxHistoryTrackerFeaturizer(max_history=4)
    training_trackers = training.load_data(
        DEFAULT_STORIES_FILE,
        default_domain,
        augmentation_factor=0
    )
    assert len(training_trackers) == 3
    (decoded, _) = featurizer.training_states_and_actions(
        training_trackers, default_domain)

    # decoded needs to be sorted
    hashed = []
    for states in decoded:
        hashed.append(json.dumps(states, sort_keys=True))
    hashed = sorted(hashed)

    assert hashed == [
        '[null, null, null, {}]',
        '[null, null, {}, '
        '{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}]',
        '[null, null, {}, '
        '{"intent_greet": 1.0, "prev_action_listen": 1.0}]',
        '[null, {}, '
        '{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}, '
        '{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_utter_greet": 1.0, "slot_name_0": 1.0}]',
        '[null, {}, '
        '{"intent_greet": 1.0, "prev_action_listen": 1.0}, '
        '{"intent_greet": 1.0, "prev_utter_greet": 1.0}]',
        '[{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}, '
        '{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_utter_greet": 1.0, "slot_name_0": 1.0}, '
        '{"intent_default": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}, '
        '{"intent_default": 1.0, '
        '"prev_utter_default": 1.0, "slot_name_0": 1.0}]',
        '[{"intent_default": 1.0, "prev_action_listen": 1.0}, '
        '{"intent_default": 1.0, "prev_utter_default": 1.0}, '
        '{"intent_goodbye": 1.0, "prev_action_listen": 1.0}, '
        '{"intent_goodbye": 1.0, "prev_utter_goodbye": 1.0}]',
        '[{"intent_greet": 1.0, "prev_action_listen": 1.0}, '
        '{"intent_greet": 1.0, "prev_utter_greet": 1.0}, '
        '{"intent_default": 1.0, "prev_action_listen": 1.0}, '
        '{"intent_default": 1.0, "prev_utter_default": 1.0}]',
        '[{"intent_greet": 1.0, "prev_utter_greet": 1.0}, '
        '{"intent_default": 1.0, "prev_action_listen": 1.0}, '
        '{"intent_default": 1.0, "prev_utter_default": 1.0}, '
        '{"intent_goodbye": 1.0, "prev_action_listen": 1.0}]',
        '[{}, {"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}, '
        '{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_utter_greet": 1.0, "slot_name_0": 1.0}, '
        '{"intent_default": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}]',
        '[{}, {"intent_greet": 1.0, "prev_action_listen": 1.0}, '
        '{"intent_greet": 1.0, "prev_utter_greet": 1.0}, '
        '{"intent_default": 1.0, "prev_action_listen": 1.0}]'
    ]


def test_domain_from_template():
    domain_file = DEFAULT_DOMAIN_PATH
    domain = TemplateDomain.load(domain_file)
    assert len(domain.intents) == 10
    assert len(domain.actions) == 6


def test_utter_templates():
    domain_file = "examples/moodbot/domain.yml"
    domain = TemplateDomain.load(domain_file)
    expected_template = {
        "text": "Hey! How are you?",
        "buttons": [{"title": "great", "payload": "great"},
                    {"title": "super sad", "payload": "super sad"}]
    }
    assert domain.random_template_for("utter_greet") == expected_template


def test_restaurant_domain_is_valid():
    # should raise no exception
    TemplateDomain.validate_domain_yaml(read_file(
            'examples/restaurantbot/restaurant_domain.yml'))


def test_custom_slot_type(tmpdir):
    domain_path = utilities.write_text_to_file(tmpdir, "domain.yml", """
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
    domain_path = utilities.write_text_to_file(tmpdir, "domain.yml", """
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


def test_domain_to_yaml():
    test_yaml = """action_factory: null
action_names:
- utter_greet
actions:
- utter_greet
config:
  store_entities_as_slots: true
entities: []
intents: []
slots: {}
templates:
  utter_greet:
  - text: hey there!"""
    domain = TemplateDomain.load_from_yaml(test_yaml)
    # python 3 and 2 are different here, python 3 will have a leading set 
    # of --- at the begining of the yml
    assert domain.as_yaml().strip().endswith(test_yaml.strip())
    domain = TemplateDomain.load_from_yaml(domain.as_yaml())
