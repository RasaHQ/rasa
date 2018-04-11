from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pytest

from rasa_core.domain import TemplateDomain
# from rasa_core.featurizers import BinaryFeaturizer
from tests import utilities
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


# TODO creation of training data changed
def test_create_train_data_no_history(default_domain):
    featurizer = BinaryFeaturizer()
    training_data = extract_training_data(
            DEFAULT_STORIES_FILE,
            default_domain,
            featurizer,
            augmentation_factor=0,
            max_history=1
    )
    assert training_data.X.shape == (11, 1, 10)
    decoded = [featurizer.decode(training_data.X[i, :, :],
                                 default_domain.input_features)
               for i in range(0, 11)]
    assert decoded == [
        [None],
        [[('intent_goodbye', 1), ('prev_utter_goodbye', 1)]],
        [[('intent_goodbye', 1), ('prev_action_listen', 1)]],
        [[('intent_default', 1), ('prev_utter_default', 1)]],
        [[('intent_default', 1), ('prev_action_listen', 1)]],
        [[('intent_default', 1), ('slot_name_0', 1),
          ('prev_utter_default', 1)]],
        [[('intent_default', 1), ('slot_name_0', 1),
          ('prev_action_listen', 1)]],
        [[('intent_greet', 1), ('prev_utter_greet', 1)]],
        [[('intent_greet', 1), ('prev_action_listen', 1)]],
        [[('intent_greet', 1), ('entity_name', 1), ('slot_name_0', 1),
          ('prev_utter_greet', 1)]],
        [[('intent_greet', 1), ('entity_name', 1), ('slot_name_0', 1),
          ('prev_action_listen', 1)]]]


def test_create_train_data_with_history(default_domain):
    featurizer = BinaryFeaturizer()
    training_data = extract_training_data(
            DEFAULT_STORIES_FILE,
            default_domain,
            featurizer,
            augmentation_factor=0,
            max_history=4
    )
    assert training_data.X.shape == (11, 4, 10)
    decoded = [featurizer.decode(training_data.X[i, :, :],
                                 default_domain.input_features)
               for i in range(0, 11)]
    assert decoded == [
        [
            None,
            [(u'intent_greet', 1), (u'prev_action_listen', 1)],
            [(u'intent_greet', 1), (u'prev_utter_greet', 1)],
            [(u'intent_default', 1), (u'prev_action_listen', 1)]],
        [
            None,
            [(u'intent_greet', 1), (u'entity_name', 1), (u'slot_name_0', 1),
             (u'prev_action_listen', 1)],
            [(u'intent_greet', 1), (u'entity_name', 1), (u'slot_name_0', 1),
             (u'prev_utter_greet', 1)],
            [(u'intent_default', 1), (u'slot_name_0', 1),
             (u'prev_action_listen', 1)]],
        [
            [(u'intent_default', 1), (u'prev_action_listen', 1)],
            [(u'intent_default', 1), (u'prev_utter_default', 1)],
            [(u'intent_goodbye', 1), (u'prev_action_listen', 1)],
            [(u'intent_goodbye', 1), (u'prev_utter_goodbye', 1)]],
        [
            [(u'intent_greet', 1), (u'prev_utter_greet', 1)],
            [(u'intent_default', 1), (u'prev_action_listen', 1)],
            [(u'intent_default', 1), (u'prev_utter_default', 1)],
            [(u'intent_goodbye', 1), (u'prev_action_listen', 1)]],
        [
            [(u'intent_greet', 1), (u'prev_action_listen', 1)],
            [(u'intent_greet', 1), (u'prev_utter_greet', 1)],
            [(u'intent_default', 1), (u'prev_action_listen', 1)],
            [(u'intent_default', 1), (u'prev_utter_default', 1)]],
        [
            [(u'intent_greet', 1), (u'entity_name', 1), (u'slot_name_0', 1),
             (u'prev_action_listen', 1)],
            [(u'intent_greet', 1), (u'entity_name', 1), (u'slot_name_0', 1),
             (u'prev_utter_greet', 1)],
            [(u'intent_default', 1), (u'slot_name_0', 1),
             (u'prev_action_listen', 1)],
            [(u'intent_default', 1), (u'slot_name_0', 1),
             (u'prev_utter_default', 1)]],
        [
            None,
            None,
            [(u'intent_greet', 1), (u'prev_action_listen', 1)],
            [(u'intent_greet', 1), (u'prev_utter_greet', 1)]],
        [
            None,
            None,
            [(u'intent_greet', 1), (u'entity_name', 1), (u'slot_name_0', 1),
             (u'prev_action_listen', 1)],
            [(u'intent_greet', 1), (u'entity_name', 1), (u'slot_name_0', 1),
             (u'prev_utter_greet', 1)]],
        [
            None, None, None, None],
        [
            None, None, None,
            [(u'intent_greet', 1), (u'prev_action_listen', 1)]],
        [
            None, None, None,
            [(u'intent_greet', 1), (u'entity_name', 1), (u'slot_name_0', 1),
             (u'prev_action_listen', 1)]]]


def test_domain_from_template():
    domain_file = DEFAULT_DOMAIN_PATH
    domain = TemplateDomain.load(domain_file)
    assert len(domain.intents) == 3
    assert len(domain.actions) == 5


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
    TemplateDomain.validate_domain_yaml(
            'examples/restaurantbot/restaurant_domain.yml')


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
