import json

import pytest
from _pytest.tmpdir import TempdirFactory

from rasa.core import training, utils
from rasa.core.domain import Domain, InvalidDomain
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.slots import TextSlot, UnfeaturizedSlot
from tests.core import utilities
from tests.core.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


async def test_create_train_data_no_history(default_domain):
    featurizer = MaxHistoryTrackerFeaturizer(max_history=1)
    training_trackers = await training.load_data(
        DEFAULT_STORIES_FILE, default_domain, augmentation_factor=0
    )

    assert len(training_trackers) == 3
    (decoded, _) = featurizer.training_states_and_actions(
        training_trackers, default_domain
    )

    # decoded needs to be sorted
    hashed = []
    for states in decoded:
        hashed.append(json.dumps(states, sort_keys=True))
    hashed = sorted(hashed, reverse=True)

    assert hashed == [
        "[{}]",
        '[{"intent_greet": 1.0, "prev_utter_greet": 1.0}]',
        '[{"intent_greet": 1.0, "prev_action_listen": 1.0}]',
        '[{"intent_goodbye": 1.0, "prev_utter_goodbye": 1.0}]',
        '[{"intent_goodbye": 1.0, "prev_action_listen": 1.0}]',
        '[{"intent_default": 1.0, "prev_utter_default": 1.0}]',
        '[{"intent_default": 1.0, "prev_utter_default": 1.0, ' '"slot_name_0": 1.0}]',
        '[{"intent_default": 1.0, "prev_action_listen": 1.0}]',
        '[{"intent_default": 1.0, "prev_action_listen": 1.0, ' '"slot_name_0": 1.0}]',
        '[{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_utter_greet": 1.0, "slot_name_0": 1.0}]',
        '[{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}]',
    ]


async def test_create_train_data_with_history(default_domain):
    featurizer = MaxHistoryTrackerFeaturizer(max_history=4)
    training_trackers = await training.load_data(
        DEFAULT_STORIES_FILE, default_domain, augmentation_factor=0
    )
    assert len(training_trackers) == 3
    (decoded, _) = featurizer.training_states_and_actions(
        training_trackers, default_domain
    )

    # decoded needs to be sorted
    hashed = []
    for states in decoded:
        hashed.append(json.dumps(states, sort_keys=True))
    hashed = sorted(hashed)

    assert hashed == [
        "[null, null, null, {}]",
        "[null, null, {}, "
        '{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}]',
        "[null, null, {}, " '{"intent_greet": 1.0, "prev_action_listen": 1.0}]',
        "[null, {}, "
        '{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_action_listen": 1.0, "slot_name_0": 1.0}, '
        '{"entity_name": 1.0, "intent_greet": 1.0, '
        '"prev_utter_greet": 1.0, "slot_name_0": 1.0}]',
        "[null, {}, "
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
        '{"intent_default": 1.0, "prev_action_listen": 1.0}]',
    ]


async def test_create_train_data_unfeaturized_entities():
    domain_file = "data/test_domains/default_unfeaturized_entities.yml"
    stories_file = "data/test_stories/stories_unfeaturized_entities.md"
    domain = Domain.load(domain_file)
    featurizer = MaxHistoryTrackerFeaturizer(max_history=1)
    training_trackers = await training.load_data(
        stories_file, domain, augmentation_factor=0
    )

    assert len(training_trackers) == 2
    (decoded, _) = featurizer.training_states_and_actions(training_trackers, domain)

    # decoded needs to be sorted
    hashed = []
    for states in decoded:
        hashed.append(json.dumps(states, sort_keys=True))
    hashed = sorted(hashed, reverse=True)

    assert hashed == [
        "[{}]",
        '[{"intent_why": 1.0, "prev_utter_default": 1.0}]',
        '[{"intent_why": 1.0, "prev_action_listen": 1.0}]',
        '[{"intent_thank": 1.0, "prev_utter_default": 1.0}]',
        '[{"intent_thank": 1.0, "prev_action_listen": 1.0}]',
        '[{"intent_greet": 1.0, "prev_utter_greet": 1.0}]',
        '[{"intent_greet": 1.0, "prev_action_listen": 1.0}]',
        '[{"intent_goodbye": 1.0, "prev_utter_goodbye": 1.0}]',
        '[{"intent_goodbye": 1.0, "prev_action_listen": 1.0}]',
        '[{"entity_name": 1.0, "intent_greet": 1.0, "prev_utter_greet": 1.0}]',
        '[{"entity_name": 1.0, "intent_greet": 1.0, "prev_action_listen": 1.0}]',
        '[{"entity_name": 1.0, "entity_other": 1.0, "intent_default": 1.0, "prev_utter_default": 1.0}]',
        '[{"entity_name": 1.0, "entity_other": 1.0, "intent_default": 1.0, "prev_action_listen": 1.0}]',
        '[{"entity_name": 1.0, "entity_other": 1.0, "entity_unrelated_recognized_entity": 1.0, "intent_ask": 1.0, "prev_utter_default": 1.0}]',
        '[{"entity_name": 1.0, "entity_other": 1.0, "entity_unrelated_recognized_entity": 1.0, "intent_ask": 1.0, "prev_action_listen": 1.0}]',
    ]


def test_domain_from_template():
    domain_file = DEFAULT_DOMAIN_PATH
    domain = Domain.load(domain_file)
    assert len(domain.intents) == 10
    assert len(domain.action_names) == 11


def test_utter_templates():
    domain_file = "examples/moodbot/domain.yml"
    domain = Domain.load(domain_file)
    expected_template = {
        "text": "Hey! How are you?",
        "buttons": [
            {"title": "great", "payload": "/mood_great"},
            {"title": "super sad", "payload": "/mood_unhappy"},
        ],
    }
    assert domain.random_template_for("utter_greet") == expected_template


def test_custom_slot_type(tmpdir):
    domain_path = utilities.write_text_to_file(
        tmpdir,
        "domain.yml",
        """
       slots:
         custom:
           type: tests.core.conftest.CustomSlot

       templates:
         utter_greet:
           - text: hey there!

       actions:
         - utter_greet """,
    )
    Domain.load(domain_path)


@pytest.mark.parametrize(
    "domain_unkown_slot_type",
    [
        """
    slots:
        custom:
         type: tests.core.conftest.Unknown

    templates:
        utter_greet:
         - text: hey there!

    actions:
        - utter_greet""",
        """
    slots:
        custom:
         type: blubblubblub

    templates:
        utter_greet:
         - text: hey there!

    actions:
        - utter_greet""",
    ],
)
def test_domain_fails_on_unknown_custom_slot_type(tmpdir, domain_unkown_slot_type):
    domain_path = utilities.write_text_to_file(
        tmpdir, "domain.yml", domain_unkown_slot_type
    )
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
    assert Domain.from_yaml(domain.as_yaml()) is not None


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


@pytest.mark.parametrize(
    "intents, intent_properties",
    [
        (
            ["greet", "goodbye"],
            {
                "greet": {"use_entities": True, "ignore_entities": []},
                "goodbye": {"use_entities": True, "ignore_entities": []},
            },
        ),
        (
            [{"greet": {"use_entities": []}}, "goodbye"],
            {
                "greet": {"use_entities": [], "ignore_entities": []},
                "goodbye": {"use_entities": True, "ignore_entities": []},
            },
        ),
        (
            [
                {
                    "greet": {
                        "triggers": "utter_goodbye",
                        "use_entities": ["entity"],
                        "ignore_entities": ["other"],
                    }
                },
                "goodbye",
            ],
            {
                "greet": {
                    "triggers": "utter_goodbye",
                    "use_entities": ["entity"],
                    "ignore_entities": ["other"],
                },
                "goodbye": {"use_entities": True, "ignore_entities": []},
            },
        ),
        (
            [
                {"greet": {"triggers": "utter_goodbye", "use_entities": None}},
                {"goodbye": {"use_entities": [], "ignore_entities": []}},
            ],
            {
                "greet": {
                    "use_entities": [],
                    "ignore_entities": [],
                    "triggers": "utter_goodbye",
                },
                "goodbye": {"use_entities": [], "ignore_entities": []},
            },
        ),
    ],
)
def test_collect_intent_properties(intents, intent_properties):
    assert Domain.collect_intent_properties(intents) == intent_properties


def test_load_domain_from_directory_tree(tmpdir_factory: TempdirFactory):
    root = tmpdir_factory.mktemp("Parent Bot")
    root_domain = {"actions": ["utter_root", "utter_root2"]}
    utils.dump_obj_as_yaml_to_file(root / "domain.yml", root_domain)

    subdirectory_1 = root / "Skill 1"
    subdirectory_1.mkdir()
    skill_1_domain = {"actions": ["utter_skill_1"]}
    utils.dump_obj_as_yaml_to_file(subdirectory_1 / "domain.yml", skill_1_domain)

    subdirectory_2 = root / "Skill 2"
    subdirectory_2.mkdir()
    skill_2_domain = {"actions": ["utter_skill_2"]}
    utils.dump_obj_as_yaml_to_file(subdirectory_2 / "domain.yml", skill_2_domain)

    subsubdirectory = subdirectory_2 / "Skill 2-1"
    subsubdirectory.mkdir()
    skill_2_1_domain = {"actions": ["utter_subskill", "utter_root"]}
    # Check if loading from `.yaml` also works
    utils.dump_obj_as_yaml_to_file(subsubdirectory / "domain.yaml", skill_2_1_domain)

    subsubdirectory_2 = subdirectory_2 / "Skill 2-2"
    subsubdirectory_2.mkdir()
    excluded_domain = {"actions": ["should not be loaded"]}
    utils.dump_obj_as_yaml_to_file(
        subsubdirectory_2 / "other_name.yaml", excluded_domain
    )

    actual = Domain.load(str(root))
    expected = [
        "utter_root",
        "utter_root2",
        "utter_skill_1",
        "utter_skill_2",
        "utter_subskill",
    ]

    assert set(actual.user_actions) == set(expected)


def test_domain_warnings():
    domain = Domain.load(DEFAULT_DOMAIN_PATH)

    warning_types = [
        "action_warnings",
        "intent_warnings",
        "entity_warnings",
        "slot_warnings",
    ]

    actions = ["action_1", "action_2"]
    intents = ["intent_1", "intent_2"]
    entities = ["entity_1", "entity_2"]
    slots = ["slot_1", "slot_2"]
    domain_warnings = domain.domain_warnings(
        intents=intents, entities=entities, actions=actions, slots=slots
    )

    # elements not found in domain should be in `in_training_data` diff
    for _type, elements in zip(warning_types, [actions, intents, entities]):
        assert set(domain_warnings[_type]["in_training_data"]) == set(elements)

    # all other domain elements should be in `in_domain` diff
    for _type, elements in zip(
        warning_types, [domain.user_actions, domain.intents, domain.entities]
    ):
        assert set(domain_warnings[_type]["in_domain"]) == set(elements)

    # fully aligned domain and elements should yield empty diff
    domain_warnings = domain.domain_warnings(
        intents=domain.intents,
        entities=domain.entities,
        actions=domain.user_actions,
        slots=[s.name for s in domain.slots],
    )

    for diff_dict in domain_warnings.values():
        assert all(not diff_set for diff_set in diff_dict.values())


def test_unfeaturized_slot_in_domain_warnings():
    # create empty domain
    domain = Domain.empty()

    # add one unfeaturized and one text slot
    unfeaturized_slot = UnfeaturizedSlot("unfeaturized_slot", "value1")
    text_slot = TextSlot("text_slot", "value2")
    domain.slots.extend([unfeaturized_slot, text_slot])

    # ensure both are in domain
    assert all(slot in domain.slots for slot in (unfeaturized_slot, text_slot))

    # text slot should appear in domain warnings, unfeaturized slot should not
    in_domain_slot_warnings = domain.domain_warnings()["slot_warnings"]["in_domain"]
    assert text_slot.name in in_domain_slot_warnings
    assert unfeaturized_slot.name not in in_domain_slot_warnings


def test_check_domain_sanity_on_invalid_domain():
    with pytest.raises(InvalidDomain):
        Domain(
            intents={},
            entities=[],
            slots=[],
            templates={},
            action_names=["random_name", "random_name"],
            form_names=[],
        )

    with pytest.raises(InvalidDomain):
        Domain(
            intents={},
            entities=[],
            slots=[TextSlot("random_name"), TextSlot("random_name")],
            templates={},
            action_names=[],
            form_names=[],
        )

    with pytest.raises(InvalidDomain):
        Domain(
            intents={},
            entities=["random_name", "random_name", "other_name", "other_name"],
            slots=[],
            templates={},
            action_names=[],
            form_names=[],
        )

    with pytest.raises(InvalidDomain):
        Domain(
            intents={},
            entities=[],
            slots=[],
            templates={},
            action_names=[],
            form_names=["random_name", "random_name"],
        )


def test_load_on_invalid_domain():
    with pytest.raises(InvalidDomain):
        Domain.load("data/test_domains/duplicate_intents.yml")

    with pytest.raises(InvalidDomain):
        Domain.load("data/test_domains/duplicate_actions.yml")

    with pytest.raises(InvalidDomain):
        Domain.load("data/test_domains/duplicate_templates.yml")

    with pytest.raises(InvalidDomain):
        Domain.load("data/test_domains/duplicate_entities.yml")

    # Currently just deprecated
    # with pytest.raises(InvalidDomain):
    #     Domain.load("data/test_domains/missing_text_for_templates.yml")


def test_clean_domain():
    domain_path = "data/test_domains/default_unfeaturized_entities.yml"
    cleaned = Domain.load(domain_path).cleaned_domain()

    expected = {
        "intents": [
            {"greet": {"use_entities": ["name"]}},
            {"default": {"ignore_entities": ["unrelated_recognized_entity"]}},
            {"goodbye": {"use_entities": []}},
            {"thank": {"use_entities": []}},
            "ask",
            {"why": {"use_entities": []}},
            "pure_intent",
        ],
        "entities": ["name", "other", "unrelated_recognized_entity"],
        "templates": {
            "utter_greet": [{"text": "hey there!"}],
            "utter_goodbye": [{"text": "goodbye :("}],
            "utter_default": [{"text": "default message"}],
        },
        "actions": ["utter_default", "utter_goodbye", "utter_greet"],
    }

    expected = Domain.from_dict(expected)
    actual = Domain.from_dict(cleaned)

    assert hash(actual) == hash(expected)
