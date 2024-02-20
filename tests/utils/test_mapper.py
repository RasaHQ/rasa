from itertools import product
from pathlib import Path
from typing import Dict, List, Text, Union

import pytest

from rasa.utils.mapper import RasaPrimitiveStorageMapper


@pytest.fixture
def primitive_names() -> Dict:
    return {
        "entities": ["age", "first_name"],
        "slots": ["age", "stargazers_count", "first_name"],
        "forms": ["personal_details_form"],
        "rules": [
            "Are you a bot",
            "Bye",
            "how many stars - successful",
            "how many stars - unsuccessful",
            "my name is",
            "personal details",
        ],
        "stories": ["how are you - feel great", "how are you - feel sad"],
        "intents": [
            "greet",
            "goodbye",
            "are_u_a_boot",
            "feel_great",
            "feel_sad",
            "how_many_stars",
            "inform",
            "start_form",
        ],
        "responses": [
            "utter_form_completed",
            "utter_ask_age",
            "utter_ask_first_name",
            "utter_hi_first_name",
            "utter_greet",
            "utter_goodbye",
            "utter_i_am_a_bot",
            "utter_how_are_you",
            "utter_great_to_hear",
            "utter_sorry_to_hear_that",
            "utter_i_have_stars",
            "utter_something_went_wrong",
        ],
        "actions": ["action_example", "action_rasa_stargazers_count"],
        "flows": ["replace_eligible_card"],
    }


@pytest.mark.parametrize(
    "domain_path",
    [
        "data/mapper/domain.yaml",
        Path("data/mapper/domain.yaml"),
    ],
)
def test_mapper_read_single_domain(
    domain_path: Union[Text, Path], primitive_names: Dict
) -> None:
    mapper = RasaPrimitiveStorageMapper(domain_path=domain_path)

    compare_domain = Path("data/mapper/domain.yaml")
    for entity in primitive_names["entities"]:
        assert (
            mapper.get_file(entity, "entities")["domain"][0].name == compare_domain.name
        )

    for intent in primitive_names["intents"]:
        assert (
            mapper.get_file(intent, "intents")["domain"][0].name == compare_domain.name
        )

    for slot in primitive_names["slots"]:
        assert mapper.get_file(slot, "slots")["domain"][0].name == compare_domain.name

    for form in primitive_names["forms"]:
        assert mapper.get_file(form, "forms")["domain"][0].name == compare_domain.name

    for response in primitive_names["responses"]:
        assert (
            mapper.get_file(response, "responses")["domain"][0].name
            == compare_domain.name
        )

    for action in primitive_names["actions"]:
        assert (
            mapper.get_file(action, "actions")["domain"][0].name == compare_domain.name
        )


@pytest.mark.parametrize(
    "domain_path",
    [
        "data/mapper/multiple_domain",
        Path("data/mapper/multiple_domain"),
    ],
)
def test_mapper_read_domain_folder(
    domain_path: Union[Text, Path], primitive_names: Dict
) -> None:
    mapper = RasaPrimitiveStorageMapper(domain_path=domain_path)

    domain_p1 = Path("data/mapper/multiple_domain/part_1.yaml")
    domain_p2 = Path("data/mapper/multiple_domain/part_2.yaml")
    for entity in primitive_names["entities"]:
        assert mapper.get_file(entity, "entities")["domain"][0].name == domain_p1.name

    for intent in primitive_names["intents"]:
        assert mapper.get_file(intent, "intents")["domain"][0].name == domain_p1.name

    for slot in primitive_names["slots"]:
        assert mapper.get_file(slot, "slots")["domain"][0].name == domain_p2.name

    for form in primitive_names["forms"]:
        assert mapper.get_file(form, "forms")["domain"][0].name == domain_p2.name


@pytest.mark.parametrize(
    "data_path",
    [
        "data/mapper/data",
        Path("data/mapper/data"),
        [
            "data/mapper/data/nlu.yml",
            "data/mapper/data/rules.yml",
            "data/mapper/data/stories.yml",
            "data/mapper/data/flows.yml",
        ],
        [
            Path("data/mapper/data/nlu.yml"),
            Path("data/mapper/data/rules.yml"),
            Path("data/mapper/data/stories.yml"),
            Path("data/mapper/data/flows.yml"),
        ],
    ],
)
def test_mapper_read_training_data(
    data_path: Union[List[Text], List[Path], Text, Path], primitive_names: Dict
) -> None:
    mapper = RasaPrimitiveStorageMapper(training_data_paths=data_path)

    nlu = Path("tests/data/mapper/data/nlu.yml")
    for intent in primitive_names["intents"]:
        assert mapper.get_file(intent, "intents")["training"][0].name == nlu.name

    rules = Path("tests/data/mapper/data/rules.yml")
    for rule in primitive_names["rules"]:
        assert mapper.get_file(rule, "rules")["training"][0].name == rules.name

    stories = Path("tests/data/mapper/data/stories.yml")
    for story in primitive_names["stories"]:
        assert mapper.get_file(story, "stories")["training"][0].name == stories.name

    flows = Path("tests/data/mapper/data/flows.yml")
    for flow in primitive_names["flows"]:
        assert mapper.get_file(flow, "flows")["training"][0].name == flows.name


def test_mapper_not_found(primitive_names: Dict) -> None:
    mapper = RasaPrimitiveStorageMapper(
        domain_path="data/mapper/domain.yaml",
        training_data_paths="data/mapper/data",
    )

    fake_primitive_type = [
        "fake_intent",
        "fake_slot",
        "fake_entity",
        "fake_form",
        "fake_rule",
        "fake_story",
        "fake_action",
        "fake_flow",
    ]

    for primitive_t in fake_primitive_type:
        assert not mapper.get_file("primitive", primitive_t)  # empty dict

    fake_primitive_name = [
        "intent_test",
        "slot_test",
        "entity_test" "form_test",
        "rule_test",
        "story_test",
        "action_test",
        "flows_test",
    ]

    for combination in product(primitive_names.keys(), fake_primitive_name):
        assert not mapper.get_file(combination[1], combination[0])
