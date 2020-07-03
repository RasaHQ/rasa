from typing import Text

import pytest

from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, UserUttered, SlotSet
from rasa.core.interpreter import RegexInterpreter
from rasa.core.training import loading
from rasa.core.training.story_reader.yaml_story_reader import YAMLStoryReader


async def test_can_read_test_story_with_slots(default_domain: Domain):
    trackers = await training.load_data(
        "data/test_yaml_stories/simple_story_with_only_end.yml",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 1

    assert trackers[0].events[-2] == SlotSet(key="name", value="peter")
    assert trackers[0].events[-1] == ActionExecuted("action_listen")


async def test_can_read_test_story_with_entities_slot_autofill(default_domain: Domain):
    trackers = await training.load_data(
        "data/test_yaml_stories/story_with_or_and_entities.yml",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 2

    assert trackers[0].events[-3] == UserUttered(
        "greet",
        intent={"name": "greet", "confidence": 1.0},
        parse_data={
            "text": "/greet",
            "intent_ranking": [{"confidence": 1.0, "name": "greet"}],
            "intent": {"confidence": 1.0, "name": "greet"},
            "entities": [],
        },
    )
    assert trackers[0].events[-2] == ActionExecuted("utter_greet")
    assert trackers[0].events[-1] == ActionExecuted("action_listen")

    assert trackers[1].events[-4] == UserUttered(
        "greet",
        intent={"name": "greet", "confidence": 1.0},
        entities=[{"entity": "name", "value": "peter"}],
        parse_data={
            "text": "/greet",
            "intent_ranking": [{"confidence": 1.0, "name": "greet"}],
            "intent": {"confidence": 1.0, "name": "greet"},
            "entities": [{"entity": "name", "value": "peter"}],
        },
    )
    assert trackers[1].events[-3] == SlotSet(key="name", value="peter")
    assert trackers[1].events[-2] == ActionExecuted("utter_greet")
    assert trackers[1].events[-1] == ActionExecuted("action_listen")


async def test_can_read_test_story_with_entities_without_value(default_domain: Domain,):
    trackers = await training.load_data(
        "data/test_yaml_stories/story_with_or_and_entities_with_no_value.yml",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 1

    assert trackers[0].events[-4] == UserUttered(
        "greet",
        intent={"name": "greet", "confidence": 1.0},
        entities=[{"entity": "name", "value": ""}],
        parse_data={
            "text": "/greet",
            "intent_ranking": [{"confidence": 1.0, "name": "greet"}],
            "intent": {"confidence": 1.0, "name": "greet"},
            "entities": [{"entity": "name", "value": ""}],
        },
    )
    assert trackers[0].events[-2] == ActionExecuted("utter_greet")
    assert trackers[0].events[-1] == ActionExecuted("action_listen")


@pytest.mark.parametrize(
    "file,is_yaml_file",
    [
        ("data/test_yaml_stories/stories.yml", True),
        ("data/test_stories/stories.md", False),
        ("data/test_yaml_stories/rules_without_stories.yml", True),
    ],
)
async def test_is_yaml_file(file: Text, is_yaml_file: bool):
    assert YAMLStoryReader.is_yaml_story_file(file) == is_yaml_file


async def test_yaml_intent_no_leading_slash_warning(default_domain: Domain):

    yaml_file = "data/test_wrong_yaml_stories/intent_without_leading_slash.yml"

    with pytest.warns(UserWarning):
        _ = await training.load_data(
            yaml_file,
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False,
        )


async def test_yaml_wrong_yaml_format_warning(default_domain: Domain):

    yaml_file = "data/test_wrong_yaml_stories/wrong_yaml.yml"

    with pytest.warns(UserWarning):
        _ = await training.load_data(
            yaml_file,
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False,
        )


async def test_read_rules(default_domain: Domain):

    yaml_file = "data/test_yaml_stories/stories_and_rules.yml"

    steps = await loading.load_data_from_files(
        [yaml_file], default_domain, RegexInterpreter()
    )

    # this file contains two rules and two stories
    # assert len(steps) == 10

    ml_steps = [s for s in steps if not s.is_rule]
    rule_steps = [s for s in steps if s.is_rule]
    print(len(ml_steps), len(rule_steps))

    assert len(ml_steps) == 9
    # assert len(rule_steps) == 3
    #
    # assert story_steps[0].block_name == "rule 1"
    # assert story_steps[1].block_name == "rule 2"
    # assert story_steps[2].block_name == "ML story 1"
    # assert story_steps[3].block_name == "rule 3"
    # assert story_steps[4].block_name == "ML story 2"
