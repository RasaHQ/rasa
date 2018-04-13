from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pytest

from rasa_core.domain import TemplateDomain
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer
from rasa_core.policies.trainer import PolicyTrainer
from tests import utilities
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


def test_create_train_data_no_history(default_domain):
    featurizer = MaxHistoryTrackerFeaturizer(None, max_history=1)
    training_trackers, _ = PolicyTrainer.extract_trackers(
            DEFAULT_STORIES_FILE,
            default_domain,
            augmentation_factor=0
    )
    assert len(training_trackers) == 3
    (decoded, _) = featurizer.training_states_and_actions(
            training_trackers, default_domain)

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
    featurizer = MaxHistoryTrackerFeaturizer(None, max_history=4)
    training_trackers, _ = PolicyTrainer.extract_trackers(
        DEFAULT_STORIES_FILE,
        default_domain,
        augmentation_factor=0
    )
    assert len(training_trackers) == 3
    (decoded, _) = featurizer.training_states_and_actions(
        training_trackers, default_domain)
    (decoded, _) = featurizer.training_states_and_actions(
        training_trackers, default_domain)

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
