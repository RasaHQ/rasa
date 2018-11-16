import json
import pytest

from rasa_core import training
from rasa_core.domain import Domain
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer
from rasa_core.utils import read_file
from rasa_core.slots import TextSlot
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
    domain = Domain.load(domain_file)
    assert len(domain.intents) == 10
    assert len(domain.action_names) == 7


def test_utter_templates():
    domain_file = "examples/moodbot/domain.yml"
    domain = Domain.load(domain_file)
    expected_template = {
        "text": "Hey! How are you?",
        "buttons": [{"title": "great", "payload": "great"},
                    {"title": "super sad", "payload": "super sad"}]
    }
    assert domain.random_template_for("utter_greet") == expected_template


def test_restaurant_domain_is_valid():
    # should raise no exception
    Domain.validate_domain_yaml(read_file(
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
    Domain.load(domain_path)


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
        Domain.load(domain_path)


def test_domain_to_yaml():
    test_yaml = """actions:
- utter_greet
config:
  store_entities_as_slots: true
entities: []
forms: []
intents: []
slots: {}
templates:
  utter_greet:
  - text: hey there!"""

    domain = Domain.from_yaml(test_yaml)
    # python 3 and 2 are different here, python 3 will have a leading set
    # of --- at the beginning of the yml
    assert domain.as_yaml().strip().endswith(test_yaml.strip())
    domain = Domain.from_yaml(domain.as_yaml())


def test_merge_yaml_domains():
    test_yaml_1 = """actions:
- utter_greet
config:
  store_entities_as_slots: true
entities: []
intents: []
slots: {}
templates:
  utter_greet:
  - text: hey there!"""

    test_yaml_2 = """actions:
- utter_greet
- utter_goodbye
config:
  store_entities_as_slots: false
entities:
- cuisine
intents:
- greet
slots:
  cuisine:
    type: text
templates:
  utter_greet:
  - text: hey you!"""

    domain_1 = Domain.from_yaml(test_yaml_1)
    domain_2 = Domain.from_yaml(test_yaml_2)
    domain = domain_1.merge(domain_2)
    # single attribute should be taken from domain_1
    assert domain.store_entities_as_slots
    # conflicts should be taken from domain_1
    assert domain.templates == {"utter_greet": [{"text": "hey there!"}]}
    # lists should be deduplicated and merged
    assert domain.intents == ["greet"]
    assert domain.entities == ["cuisine"]
    assert isinstance(domain.slots[0], TextSlot)
    assert domain.slots[0].name == "cuisine"
    assert sorted(domain.user_actions) == sorted(["utter_greet", "utter_goodbye"])

    domain = domain_1.merge(domain_2, override=True)
    # single attribute should be taken from domain_2
    assert not domain.store_entities_as_slots
    # conflicts should take value from domain_2
    assert domain.templates == {"utter_greet": [{"text": "hey you!"}]}
