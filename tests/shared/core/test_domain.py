import copy
import json
from pathlib import Path
from typing import Dict, List, Text, Any, Union, Set

import pytest

from rasa.shared.exceptions import YamlSyntaxException
import rasa.shared.utils.io
from rasa.shared.constants import DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES
from rasa.core import training, utils
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.shared.core.slots import InvalidSlotTypeException, TextSlot
from rasa.shared.core.constants import (
    DEFAULT_INTENTS,
    SLOT_LISTED_ITEMS,
    SLOT_LAST_OBJECT,
    SLOT_LAST_OBJECT_TYPE,
    DEFAULT_KNOWLEDGE_BASE_ACTION,
    ENTITY_LABEL_SEPARATOR,
)
from rasa.shared.core.domain import (
    InvalidDomain,
    SessionConfig,
    ENTITY_ROLES_KEY,
    USED_ENTITIES_KEY,
    USE_ENTITIES_KEY,
    IGNORE_ENTITIES_KEY,
    State,
    Domain,
    KEY_FORMS,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_DOMAIN_PATH_WITH_SLOTS_AND_NO_ACTIONS,
    DEFAULT_STORIES_FILE,
)


def test_slots_states_before_user_utterance(default_domain: Domain):
    featurizer = MaxHistoryTrackerFeaturizer()
    tracker = DialogueStateTracker.from_events(
        "bla",
        evts=[
            SlotSet(default_domain.slots[0].name, "some_value"),
            ActionExecuted("utter_default"),
        ],
        slots=default_domain.slots,
    )
    trackers_as_states, _ = featurizer.training_states_and_actions(
        [tracker], default_domain
    )
    expected_states = [[{"slots": {"name": (1.0,)}}]]
    assert trackers_as_states == expected_states


async def test_create_train_data_no_history(default_domain: Domain):
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
        '[{"prev_action": {"action_name": "utter_greet"}, "user": {"intent": "greet"}}]',
        '[{"prev_action": {"action_name": "utter_greet"}, "slots": {"name": [1.0]}, "user": {"entities": ["name"], "intent": "greet"}}]',
        '[{"prev_action": {"action_name": "utter_goodbye"}, "user": {"intent": "goodbye"}}]',
        '[{"prev_action": {"action_name": "utter_default"}, "user": {"intent": "default"}}]',
        '[{"prev_action": {"action_name": "utter_default"}, "slots": {"name": [1.0]}, "user": {"intent": "default"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "greet"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "goodbye"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "default"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "slots": {"name": [1.0]}, "user": {"intent": "default"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "slots": {"name": [1.0]}, "user": {"entities": ["name"], "intent": "greet"}}]',
    ]


async def test_create_train_data_with_history(default_domain: Domain):
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
        '[{"prev_action": {"action_name": "action_listen"}, "slots": {"name": [1.0]}, "user": {"entities": ["name"], "intent": "greet"}}, {"prev_action": {"action_name": "utter_greet"}, "slots": {"name": [1.0]}, "user": {"entities": ["name"], "intent": "greet"}}, {"prev_action": {"action_name": "action_listen"}, "slots": {"name": [1.0]}, "user": {"intent": "default"}}, {"prev_action": {"action_name": "utter_default"}, "slots": {"name": [1.0]}, "user": {"intent": "default"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "default"}}, {"prev_action": {"action_name": "utter_default"}, "user": {"intent": "default"}}, {"prev_action": {"action_name": "action_listen"}, "user": {"intent": "goodbye"}}, {"prev_action": {"action_name": "utter_goodbye"}, "user": {"intent": "goodbye"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "greet"}}, {"prev_action": {"action_name": "utter_greet"}, "user": {"intent": "greet"}}, {"prev_action": {"action_name": "action_listen"}, "user": {"intent": "default"}}, {"prev_action": {"action_name": "utter_default"}, "user": {"intent": "default"}}]',
        '[{"prev_action": {"action_name": "utter_greet"}, "user": {"intent": "greet"}}, {"prev_action": {"action_name": "action_listen"}, "user": {"intent": "default"}}, {"prev_action": {"action_name": "utter_default"}, "user": {"intent": "default"}}, {"prev_action": {"action_name": "action_listen"}, "user": {"intent": "goodbye"}}]',
        '[{}, {"prev_action": {"action_name": "action_listen"}, "slots": {"name": [1.0]}, "user": {"entities": ["name"], "intent": "greet"}}, {"prev_action": {"action_name": "utter_greet"}, "slots": {"name": [1.0]}, "user": {"entities": ["name"], "intent": "greet"}}, {"prev_action": {"action_name": "action_listen"}, "slots": {"name": [1.0]}, "user": {"intent": "default"}}]',
        '[{}, {"prev_action": {"action_name": "action_listen"}, "slots": {"name": [1.0]}, "user": {"entities": ["name"], "intent": "greet"}}, {"prev_action": {"action_name": "utter_greet"}, "slots": {"name": [1.0]}, "user": {"entities": ["name"], "intent": "greet"}}]',
        '[{}, {"prev_action": {"action_name": "action_listen"}, "slots": {"name": [1.0]}, "user": {"entities": ["name"], "intent": "greet"}}]',
        '[{}, {"prev_action": {"action_name": "action_listen"}, "user": {"intent": "greet"}}, {"prev_action": {"action_name": "utter_greet"}, "user": {"intent": "greet"}}, {"prev_action": {"action_name": "action_listen"}, "user": {"intent": "default"}}]',
        '[{}, {"prev_action": {"action_name": "action_listen"}, "user": {"intent": "greet"}}, {"prev_action": {"action_name": "utter_greet"}, "user": {"intent": "greet"}}]',
        '[{}, {"prev_action": {"action_name": "action_listen"}, "user": {"intent": "greet"}}]',
        "[{}]",
    ]


def check_for_too_many_entities_and_remove_them(state: State) -> State:
    # we ignore entities where there are > 1 of them:
    # entities come from dictionary keys; as a result, they are stored
    # in different order in the tuple which makes the test unstable
    if (
        state.get("user")
        and state.get("user", {}).get("entities")
        and len(state.get("user").get("entities")) > 1
    ):
        state.get("user")["entities"] = ()
    return state


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
        new_states = [
            check_for_too_many_entities_and_remove_them(state) for state in states
        ]

        hashed.append(json.dumps(new_states, sort_keys=True))
    hashed = sorted(hashed, reverse=True)

    assert hashed == [
        "[{}]",
        '[{"prev_action": {"action_name": "utter_greet"}, "user": {"intent": "greet"}}]',
        '[{"prev_action": {"action_name": "utter_greet"}, "user": {"entities": ["name"], "intent": "greet"}}]',
        '[{"prev_action": {"action_name": "utter_goodbye"}, "user": {"intent": "goodbye"}}]',
        '[{"prev_action": {"action_name": "utter_default"}, "user": {"intent": "why"}}]',
        '[{"prev_action": {"action_name": "utter_default"}, "user": {"intent": "thank"}}]',
        '[{"prev_action": {"action_name": "utter_default"}, "user": {"entities": [], "intent": "default"}}]',
        '[{"prev_action": {"action_name": "utter_default"}, "user": {"entities": [], "intent": "ask"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "why"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "thank"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "greet"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "goodbye"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"entities": [], "intent": "default"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"entities": [], "intent": "ask"}}]',
        '[{"prev_action": {"action_name": "action_listen"}, "user": {"entities": ["name"], "intent": "greet"}}]',
    ]


def test_domain_from_template():
    domain_file = DEFAULT_DOMAIN_PATH_WITH_SLOTS
    domain = Domain.load(domain_file)

    assert not domain.is_empty()
    assert len(domain.intents) == 10 + len(DEFAULT_INTENTS)
    assert len(domain.action_names) == 15


def test_avoid_action_repetition():
    domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
    domain_with_no_actions = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS_AND_NO_ACTIONS)

    assert not domain.is_empty() and not domain_with_no_actions.is_empty()
    assert len(domain.intents) == len(domain_with_no_actions.intents)
    assert len(domain.action_names) == len(domain_with_no_actions.action_names)


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


def test_custom_slot_type(tmpdir: Path):
    domain_path = str(tmpdir / "domain.yml")
    rasa.shared.utils.io.write_text_file(
        """
       slots:
         custom:
           type: tests.core.conftest.CustomSlot

       responses:
         utter_greet:
           - text: hey there! """,
        domain_path,
    )
    Domain.load(domain_path)


@pytest.mark.parametrize(
    "domain_unkown_slot_type",
    [
        """
    slots:
        custom:
         type: tests.core.conftest.Unknown

    responses:
        utter_greet:
         - text: hey there!""",
        """
    slots:
        custom:
         type: blubblubblub

    responses:
        utter_greet:
         - text: hey there!""",
    ],
)
def test_domain_fails_on_unknown_custom_slot_type(tmpdir, domain_unkown_slot_type):
    domain_path = str(tmpdir / "domain.yml")
    rasa.shared.utils.io.write_text_file(domain_unkown_slot_type, domain_path)
    with pytest.raises(InvalidSlotTypeException):
        Domain.load(domain_path)


def test_domain_to_dict():
    test_yaml = """
    actions:
    - action_save_world
    config:
      store_entities_as_slots: true
    entities: []
    forms: []
    intents: []
    responses:
      utter_greet:
      - text: hey there!
    session_config:
      carry_over_slots_to_new_session: true
      session_expiration_time: 60
    slots: {}"""

    domain_as_dict = Domain.from_yaml(test_yaml).as_dict()

    assert domain_as_dict == {
        "actions": ["action_save_world"],
        "config": {"store_entities_as_slots": True},
        "entities": [],
        "forms": {},
        "intents": [],
        "e2e_actions": [],
        "responses": {"utter_greet": [{"text": "hey there!"}]},
        "session_config": {
            "carry_over_slots_to_new_session": True,
            "session_expiration_time": 60,
        },
        "slots": {},
    }


def test_domain_to_yaml():
    test_yaml = f"""
version: '2.0'
actions:
- action_save_world
config:
  store_entities_as_slots: true
e2e_actions: []
entities: []
forms: {{}}
intents: []
responses:
  utter_greet:
  - text: hey there!
session_config:
  carry_over_slots_to_new_session: true
  session_expiration_time: {DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES}
slots: {{}}
"""

    with pytest.warns(None) as record:
        domain = Domain.from_yaml(test_yaml)
        actual_yaml = domain.as_yaml()

    assert not record

    expected = rasa.shared.utils.io.read_yaml(test_yaml)
    actual = rasa.shared.utils.io.read_yaml(actual_yaml)
    assert actual == expected


def test_merge_yaml_domains():
    test_yaml_1 = """config:
  store_entities_as_slots: true
entities: []
intents: []
slots: {}
responses:
  utter_greet:
  - text: hey there!"""

    test_yaml_2 = """config:
  store_entities_as_slots: false
session_config:
    session_expiration_time: 20
    carry_over_slots: true
entities:
- cuisine
intents:
- greet
slots:
  cuisine:
    type: text
responses:
  utter_goodbye:
  - text: bye!
  utter_greet:
  - text: hey you!"""

    domain_1 = Domain.from_yaml(test_yaml_1)
    domain_2 = Domain.from_yaml(test_yaml_2)
    domain = domain_1.merge(domain_2)
    # single attribute should be taken from domain_1
    assert domain.store_entities_as_slots
    # conflicts should be taken from domain_1
    assert domain.templates == {
        "utter_greet": [{"text": "hey there!"}],
        "utter_goodbye": [{"text": "bye!"}],
    }
    # lists should be deduplicated and merged
    assert domain.intents == sorted(["greet", *DEFAULT_INTENTS])
    assert domain.entities == ["cuisine"]
    assert isinstance(domain.slots[0], TextSlot)
    assert domain.slots[0].name == "cuisine"
    assert sorted(domain.user_actions) == sorted(["utter_greet", "utter_goodbye"])
    assert domain.session_config == SessionConfig(20, True)

    domain = domain_1.merge(domain_2, override=True)
    # single attribute should be taken from domain_2
    assert not domain.store_entities_as_slots
    # conflicts should take value from domain_2
    assert domain.templates == {
        "utter_greet": [{"text": "hey you!"}],
        "utter_goodbye": [{"text": "bye!"}],
    }
    assert domain.session_config == SessionConfig(20, True)


@pytest.mark.parametrize("default_intent", DEFAULT_INTENTS)
def test_merge_yaml_domains_with_default_intents(default_intent: Text):
    test_yaml_1 = """intents: []"""

    # this domain contains an overridden default intent
    test_yaml_2 = f"""intents:
- greet
- {default_intent}"""

    domain_1 = Domain.from_yaml(test_yaml_1)
    domain_2 = Domain.from_yaml(test_yaml_2)
    domain = domain_1.merge(domain_2)

    # check that the default intents were merged correctly
    assert default_intent in domain.intents
    assert domain.intents == sorted(["greet", *DEFAULT_INTENTS])

    # ensure that the default intent is contain the domain's dictionary dump
    assert list(domain.as_dict()["intents"][1].keys())[0] == default_intent


def test_merge_session_config_if_first_is_not_default():
    yaml1 = """
session_config:
    session_expiration_time: 20
    carry_over_slots: true"""

    yaml2 = """
 session_config:
    session_expiration_time: 40
    carry_over_slots: true
    """

    domain1 = Domain.from_yaml(yaml1)
    domain2 = Domain.from_yaml(yaml2)

    merged = domain1.merge(domain2)
    assert merged.session_config == SessionConfig(20, True)

    merged = domain1.merge(domain2, override=True)
    assert merged.session_config == SessionConfig(40, True)


def test_merge_with_empty_domain():
    domain = Domain.from_yaml(
        """config:
  store_entities_as_slots: false
session_config:
    session_expiration_time: 20
    carry_over_slots: true
entities:
- cuisine
intents:
- greet
slots:
  cuisine:
    type: text
responses:
  utter_goodbye:
  - text: bye!
  utter_greet:
  - text: hey you!"""
    )

    merged = Domain.empty().merge(domain)

    assert merged.as_dict() == domain.as_dict()


def test_merge_with_empty_other_domain():
    domain = Domain.from_yaml(
        """config:
  store_entities_as_slots: false
session_config:
    session_expiration_time: 20
    carry_over_slots: true
entities:
- cuisine
intents:
- greet
slots:
  cuisine:
    type: text
responses:
  utter_goodbye:
  - text: bye!
  utter_greet:
  - text: hey you!"""
    )

    merged = domain.merge(Domain.empty(), override=True)

    assert merged.as_dict() == domain.as_dict()


def test_merge_domain_with_forms():
    test_yaml_1 = """
    forms:
    # Old style form definitions (before RulePolicy)
    - my_form
    - my_form2
    """

    test_yaml_2 = """
    forms:
      my_form3:
        slot1:
        - type: from_text
    """

    domain_1 = Domain.from_yaml(test_yaml_1)
    domain_2 = Domain.from_yaml(test_yaml_2)
    domain = domain_1.merge(domain_2)

    expected_number_of_forms = 3
    assert len(domain.form_names) == expected_number_of_forms
    assert len(domain.forms) == expected_number_of_forms


@pytest.mark.parametrize(
    "intents, entities, roles, groups, intent_properties",
    [
        (
            ["greet", "goodbye"],
            ["entity", "other", "third"],
            {"entity": ["role-1", "role-2"]},
            {},
            {
                "greet": {
                    USED_ENTITIES_KEY: [
                        "entity",
                        f"entity{ENTITY_LABEL_SEPARATOR}role-1",
                        f"entity{ENTITY_LABEL_SEPARATOR}role-2",
                        "other",
                        "third",
                    ]
                },
                "goodbye": {
                    USED_ENTITIES_KEY: [
                        "entity",
                        f"entity{ENTITY_LABEL_SEPARATOR}role-1",
                        f"entity{ENTITY_LABEL_SEPARATOR}role-2",
                        "other",
                        "third",
                    ]
                },
            },
        ),
        (
            [{"greet": {USE_ENTITIES_KEY: []}}, "goodbye"],
            ["entity", "other", "third"],
            {},
            {"other": ["1", "2"]},
            {
                "greet": {USED_ENTITIES_KEY: []},
                "goodbye": {
                    USED_ENTITIES_KEY: [
                        "entity",
                        "other",
                        f"other{ENTITY_LABEL_SEPARATOR}1",
                        f"other{ENTITY_LABEL_SEPARATOR}2",
                        "third",
                    ]
                },
            },
        ),
        (
            [
                {
                    "greet": {
                        "triggers": "utter_goodbye",
                        USE_ENTITIES_KEY: ["entity"],
                        IGNORE_ENTITIES_KEY: ["other"],
                    }
                },
                "goodbye",
            ],
            ["entity", "other", "third"],
            {"entity": ["role"], "other": ["role"]},
            {},
            {
                "greet": {
                    "triggers": "utter_goodbye",
                    USED_ENTITIES_KEY: [
                        "entity",
                        f"entity{ENTITY_LABEL_SEPARATOR}role",
                    ],
                },
                "goodbye": {
                    USED_ENTITIES_KEY: [
                        "entity",
                        f"entity{ENTITY_LABEL_SEPARATOR}role",
                        "other",
                        f"other{ENTITY_LABEL_SEPARATOR}role",
                        "third",
                    ]
                },
            },
        ),
        (
            [
                {"greet": {"triggers": "utter_goodbye", USE_ENTITIES_KEY: None}},
                {"goodbye": {USE_ENTITIES_KEY: [], IGNORE_ENTITIES_KEY: []}},
            ],
            ["entity", "other", "third"],
            {},
            {},
            {
                "greet": {USED_ENTITIES_KEY: [], "triggers": "utter_goodbye"},
                "goodbye": {USED_ENTITIES_KEY: []},
            },
        ),
        (
            [
                "greet",
                "goodbye",
                {"chitchat": {"is_retrieval_intent": True, "use_entities": None}},
            ],
            ["entity", "other", "third"],
            {},
            {},
            {
                "greet": {USED_ENTITIES_KEY: ["entity", "other", "third"]},
                "goodbye": {USED_ENTITIES_KEY: ["entity", "other", "third"]},
                "chitchat": {USED_ENTITIES_KEY: [], "is_retrieval_intent": True},
            },
        ),
    ],
)
def test_collect_intent_properties(
    intents: Union[Set[Text], List[Union[Text, Dict[Text, Any]]]],
    entities: List[Text],
    roles: Dict[Text, List[Text]],
    groups: Dict[Text, List[Text]],
    intent_properties: Dict[Text, Dict[Text, Union[bool, List]]],
):
    Domain._add_default_intents(intent_properties, entities, roles, groups)

    assert (
        Domain.collect_intent_properties(intents, entities, roles, groups)
        == intent_properties
    )


def test_load_domain_from_directory_tree(tmp_path: Path):
    root_domain = {"actions": ["utter_root", "utter_root2"]}
    utils.dump_obj_as_yaml_to_file(tmp_path / "domain_pt1.yml", root_domain)

    subdirectory_1 = tmp_path / "Skill 1"
    subdirectory_1.mkdir()
    skill_1_domain = {"actions": ["utter_skill_1"]}
    utils.dump_obj_as_yaml_to_file(subdirectory_1 / "domain_pt2.yml", skill_1_domain)

    subdirectory_2 = tmp_path / "Skill 2"
    subdirectory_2.mkdir()
    skill_2_domain = {"actions": ["utter_skill_2"]}
    utils.dump_obj_as_yaml_to_file(subdirectory_2 / "domain_pt3.yml", skill_2_domain)

    subsubdirectory = subdirectory_2 / "Skill 2-1"
    subsubdirectory.mkdir()
    skill_2_1_domain = {"actions": ["utter_subskill", "utter_root"]}
    # Check if loading from `.yaml` also works
    utils.dump_obj_as_yaml_to_file(
        subsubdirectory / "domain_pt4.yaml", skill_2_1_domain
    )

    actual = Domain.load(str(tmp_path))
    expected = [
        "utter_root",
        "utter_root2",
        "utter_skill_1",
        "utter_skill_2",
        "utter_subskill",
    ]

    assert set(actual.user_actions) == set(expected)


def test_domain_warnings():
    domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)

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
    unfeaturized_slot = TextSlot(
        "unfeaturized_slot", "value1", influence_conversation=False
    )
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
            forms={},
        )

    with pytest.raises(InvalidDomain):
        Domain(
            intents={},
            entities=[],
            slots=[TextSlot("random_name"), TextSlot("random_name")],
            templates={},
            action_names=[],
            forms={},
        )

    with pytest.raises(InvalidDomain):
        Domain(
            intents={},
            entities=["random_name", "random_name", "other_name", "other_name"],
            slots=[],
            templates={},
            action_names=[],
            forms={},
        )

    with pytest.raises(InvalidDomain):
        Domain(
            intents={},
            entities=[],
            slots=[],
            templates={},
            action_names=[],
            forms=["random_name", "random_name"],
        )


def test_load_on_invalid_domain_duplicate_intents():
    with pytest.raises(InvalidDomain):
        Domain.load("data/test_domains/duplicate_intents.yml")


def test_load_on_invalid_domain_duplicate_actions():
    with pytest.raises(InvalidDomain):
        Domain.load("data/test_domains/duplicate_actions.yml")


def test_load_on_invalid_domain_duplicate_templates():
    with pytest.raises(YamlSyntaxException):
        Domain.load("data/test_domains/duplicate_templates.yml")


def test_load_on_invalid_domain_duplicate_entities():
    with pytest.raises(InvalidDomain):
        Domain.load("data/test_domains/duplicate_entities.yml")


def test_load_domain_with_entity_roles_groups():
    domain = Domain.load("data/test_domains/travel_form.yml")

    assert domain.entities is not None
    assert "GPE" in domain.entities
    assert "name" in domain.entities
    assert "name" not in domain.roles
    assert "GPE" in domain.roles
    assert "origin" in domain.roles["GPE"]
    assert "destination" in domain.roles["GPE"]


def test_is_empty():
    assert Domain.empty().is_empty()


def test_transform_intents_for_file_default():
    domain_path = "data/test_domains/default_unfeaturized_entities.yml"
    domain = Domain.load(domain_path)
    transformed = domain._transform_intents_for_file()

    expected = [
        {"greet": {USE_ENTITIES_KEY: ["name"]}},
        {"default": {IGNORE_ENTITIES_KEY: ["unrelated_recognized_entity"]}},
        {"goodbye": {USE_ENTITIES_KEY: []}},
        {"thank": {USE_ENTITIES_KEY: []}},
        {"ask": {USE_ENTITIES_KEY: True}},
        {"why": {USE_ENTITIES_KEY: []}},
        {"pure_intent": {USE_ENTITIES_KEY: True}},
    ]

    assert transformed == expected


def test_transform_intents_for_file_with_mapping():
    domain_path = "data/test_domains/default_with_mapping.yml"
    domain = Domain.load(domain_path)
    transformed = domain._transform_intents_for_file()

    expected = [
        {"greet": {"triggers": "utter_greet", USE_ENTITIES_KEY: True}},
        {"default": {"triggers": "utter_default", USE_ENTITIES_KEY: True}},
        {"goodbye": {USE_ENTITIES_KEY: True}},
    ]

    assert transformed == expected


def test_transform_intents_for_file_with_entity_roles_groups():
    domain_path = "data/test_domains/travel_form.yml"
    domain = Domain.load(domain_path)
    transformed = domain._transform_intents_for_file()

    expected = [
        {"inform": {USE_ENTITIES_KEY: ["GPE"]}},
        {"greet": {USE_ENTITIES_KEY: ["name"]}},
    ]

    assert transformed == expected


def test_transform_entities_for_file_default():
    domain_path = "data/test_domains/travel_form.yml"
    domain = Domain.load(domain_path)
    transformed = domain._transform_entities_for_file()

    expected = [{"GPE": {ENTITY_ROLES_KEY: ["destination", "origin"]}}, "name"]

    assert transformed == expected


def test_clean_domain_for_file():
    domain_path = "data/test_domains/default_unfeaturized_entities.yml"
    cleaned = Domain.load(domain_path).cleaned_domain()

    expected = {
        "intents": [
            {"greet": {USE_ENTITIES_KEY: ["name"]}},
            {"default": {IGNORE_ENTITIES_KEY: ["unrelated_recognized_entity"]}},
            {"goodbye": {USE_ENTITIES_KEY: []}},
            {"thank": {USE_ENTITIES_KEY: []}},
            "ask",
            {"why": {USE_ENTITIES_KEY: []}},
            "pure_intent",
        ],
        "entities": ["name", "unrelated_recognized_entity", "other"],
        "responses": {
            "utter_greet": [{"text": "hey there!"}],
            "utter_goodbye": [{"text": "goodbye :("}],
            "utter_default": [{"text": "default message"}],
        },
        "session_config": {
            "carry_over_slots_to_new_session": True,
            "session_expiration_time": DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
        },
    }

    assert cleaned == expected


def test_add_knowledge_base_slots(default_domain: Domain):
    # don't modify default domain as it is used in other tests
    test_domain = copy.deepcopy(default_domain)

    test_domain.action_names.append(DEFAULT_KNOWLEDGE_BASE_ACTION)

    slot_names = [s.name for s in test_domain.slots]

    assert SLOT_LISTED_ITEMS not in slot_names
    assert SLOT_LAST_OBJECT not in slot_names
    assert SLOT_LAST_OBJECT_TYPE not in slot_names

    test_domain.add_knowledge_base_slots()

    slot_names = [s.name for s in test_domain.slots]

    assert SLOT_LISTED_ITEMS in slot_names
    assert SLOT_LAST_OBJECT in slot_names
    assert SLOT_LAST_OBJECT_TYPE in slot_names


@pytest.mark.parametrize(
    "input_domain, expected_session_expiration_time, expected_carry_over_slots",
    [
        (
            f"""session_config:
    session_expiration_time: {DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES}
    carry_over_slots_to_new_session: true""",
            DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
            True,
        ),
        ("", DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES, True),
        (
            """session_config:
    carry_over_slots_to_new_session: false""",
            DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES,
            False,
        ),
        (
            """session_config:
    session_expiration_time: 20.2
    carry_over_slots_to_new_session: False""",
            20.2,
            False,
        ),
        ("""session_config: {}""", DEFAULT_SESSION_EXPIRATION_TIME_IN_MINUTES, True),
    ],
)
def test_session_config(
    input_domain,
    expected_session_expiration_time: float,
    expected_carry_over_slots: bool,
):
    domain = Domain.from_yaml(input_domain)
    assert (
        domain.session_config.session_expiration_time
        == expected_session_expiration_time
    )
    assert domain.session_config.carry_over_slots == expected_carry_over_slots


def test_domain_as_dict_with_session_config():
    session_config = SessionConfig(123, False)
    domain = Domain.empty()
    domain.session_config = session_config

    serialized = domain.as_dict()
    deserialized = Domain.from_dict(serialized)

    assert deserialized.session_config == session_config


@pytest.mark.parametrize(
    "session_config, enabled",
    [
        (SessionConfig(0, True), False),
        (SessionConfig(1, True), True),
        (SessionConfig(-1, False), False),
    ],
)
def test_are_sessions_enabled(session_config: SessionConfig, enabled: bool):
    assert session_config.are_sessions_enabled() == enabled


def test_domain_from_dict_does_not_change_input():
    input_before = {
        "intents": [
            {"greet": {USE_ENTITIES_KEY: ["name"]}},
            {"default": {IGNORE_ENTITIES_KEY: ["unrelated_recognized_entity"]}},
            {"goodbye": {USE_ENTITIES_KEY: None}},
            {"thank": {USE_ENTITIES_KEY: False}},
            {"ask": {USE_ENTITIES_KEY: True}},
            {"why": {USE_ENTITIES_KEY: []}},
            "pure_intent",
        ],
        "entities": ["name", "unrelated_recognized_entity", "other"],
        "slots": {"name": {"type": "text"}},
        "responses": {
            "utter_greet": [{"text": "hey there {name}!"}],
            "utter_goodbye": [{"text": "goodbye 😢"}, {"text": "bye bye 😢"}],
            "utter_default": [{"text": "default message"}],
        },
    }

    input_after = copy.deepcopy(input_before)
    Domain.from_dict(input_after)

    assert input_after == input_before


@pytest.mark.parametrize(
    "domain", [{}, {"intents": DEFAULT_INTENTS}, {"intents": [DEFAULT_INTENTS[0]]}]
)
def test_add_default_intents(domain: Dict):
    domain = Domain.from_dict(domain)

    assert all(intent_name in domain.intents for intent_name in DEFAULT_INTENTS)


def test_domain_deepcopy():
    domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
    new_domain = copy.deepcopy(domain)

    assert isinstance(new_domain, Domain)

    # equalities
    assert new_domain.intent_properties == domain.intent_properties
    assert new_domain.overridden_default_intents == domain.overridden_default_intents
    assert new_domain.entities == domain.entities
    assert new_domain.forms == domain.forms
    assert new_domain.form_names == domain.form_names
    assert new_domain.templates == domain.templates
    assert new_domain.action_texts == domain.action_texts
    assert new_domain.session_config == domain.session_config
    assert new_domain._custom_actions == domain._custom_actions
    assert new_domain.user_actions == domain.user_actions
    assert new_domain.action_names == domain.action_names
    assert new_domain.store_entities_as_slots == domain.store_entities_as_slots

    # not the same objects
    assert new_domain is not domain
    assert new_domain.intent_properties is not domain.intent_properties
    assert (
        new_domain.overridden_default_intents is not domain.overridden_default_intents
    )
    assert new_domain.entities is not domain.entities
    assert new_domain.forms is not domain.forms
    assert new_domain.form_names is not domain.form_names
    assert new_domain.slots is not domain.slots
    assert new_domain.templates is not domain.templates
    assert new_domain.action_texts is not domain.action_texts
    assert new_domain.session_config is not domain.session_config
    assert new_domain._custom_actions is not domain._custom_actions
    assert new_domain.user_actions is not domain.user_actions
    assert new_domain.action_names is not domain.action_names


@pytest.mark.parametrize(
    "template_key, validation",
    [("utter_chitchat/faq", True), ("utter_chitchat", False)],
)
def test_is_retrieval_intent_template(template_key, validation):
    domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
    assert domain.is_retrieval_intent_template((template_key, [{}])) == validation


def test_retrieval_intent_template_seggregation():
    domain = Domain.load("data/test_domains/mixed_retrieval_intents.yml")
    assert domain.templates != domain.retrieval_intent_templates
    assert domain.templates and domain.retrieval_intent_templates
    assert list(domain.retrieval_intent_templates.keys()) == [
        "utter_chitchat/ask_weather",
        "utter_chitchat/ask_name",
    ]


def test_get_featurized_entities():
    domain = Domain.load("data/test_domains/travel_form.yml")

    user_uttered = UserUttered(
        text="Hello, I am going to London",
        intent={"name": "greet", "confidence": 1.0},
        entities=[{"entity": "GPE", "value": "London", "role": "destination"}],
    )

    featurized_entities = domain._get_featurized_entities(user_uttered)

    assert featurized_entities == set()

    user_uttered = UserUttered(
        text="I am going to London",
        intent={"inform": "greet", "confidence": 1.0},
        entities=[{"entity": "GPE", "value": "London", "role": "destination"}],
    )

    featurized_entities = domain._get_featurized_entities(user_uttered)

    assert featurized_entities == {"GPE", f"GPE{ENTITY_LABEL_SEPARATOR}destination"}


@pytest.mark.parametrize(
    "domain_as_dict",
    [
        # No forms
        {KEY_FORMS: {}},
        # Deprecated but still support form syntax
        {KEY_FORMS: ["my form", "other form"]},
        # No slot mappings
        {KEY_FORMS: {"my_form": None}},
        {KEY_FORMS: {"my_form": {}}},
        # Valid slot mappings
        {
            KEY_FORMS: {
                "my_form": {"slot_x": [{"type": "from_entity", "entity": "name"}]}
            }
        },
        {KEY_FORMS: {"my_form": {"slot_x": [{"type": "from_intent", "value": 5}]}}},
        {
            KEY_FORMS: {
                "my_form": {"slot_x": [{"type": "from_intent", "value": "some value"}]}
            }
        },
        {KEY_FORMS: {"my_form": {"slot_x": [{"type": "from_intent", "value": False}]}}},
        {
            KEY_FORMS: {
                "my_form": {"slot_x": [{"type": "from_trigger_intent", "value": 5}]}
            }
        },
        {
            KEY_FORMS: {
                "my_form": {
                    "slot_x": [{"type": "from_trigger_intent", "value": "some value"}]
                }
            }
        },
        {KEY_FORMS: {"my_form": {"slot_x": [{"type": "from_text"}]}}},
    ],
)
def test_valid_slot_mappings(domain_as_dict: Dict[Text, Any]):
    Domain.from_dict(domain_as_dict)


@pytest.mark.parametrize(
    "domain_as_dict",
    [
        # Wrong type for slot names
        {KEY_FORMS: {"my_form": []}},
        {KEY_FORMS: {"my_form": 5}},
        # Slot mappings not defined as list
        {KEY_FORMS: {"my_form": {"slot1": {}}}},
        # Unknown mapping
        {KEY_FORMS: {"my_form": {"slot1": [{"type": "my slot mapping"}]}}},
        # Mappings with missing keys
        {
            KEY_FORMS: {
                "my_form": {"slot1": [{"type": "from_entity", "intent": "greet"}]}
            }
        },
        {KEY_FORMS: {"my_form": {"slot1": [{"type": "from_intent"}]}}},
        {KEY_FORMS: {"my_form": {"slot1": [{"type": "from_intent", "value": None}]}}},
        {KEY_FORMS: {"my_form": {"slot1": [{"type": "from_trigger_intent"}]}}},
        {
            KEY_FORMS: {
                "my_form": {"slot1": [{"type": "from_trigger_intent", "value": None}]}
            }
        },
    ],
)
def test_form_invalid_mappings(domain_as_dict: Dict[Text, Any]):
    with pytest.raises(InvalidDomain):
        Domain.from_dict(domain_as_dict)
