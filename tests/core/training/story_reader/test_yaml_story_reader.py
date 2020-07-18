from typing import Text

import pytest

from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.training import loading
from rasa.core.events import ActionExecuted, UserUttered, SlotSet, Form
from rasa.core.interpreter import RegexInterpreter
from rasa.core.training.story_reader.yaml_story_reader import YAMLStoryReader
from rasa.utils import io as io_utils


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


async def test_yaml_intent_with_leading_slash_warning(default_domain: Domain):
    yaml_file = "data/test_wrong_yaml_stories/intent_with_leading_slash.yml"

    with pytest.warns(UserWarning):
        tracker = await training.load_data(
            yaml_file,
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False,
        )

    assert tracker[0].latest_message == UserUttered("simple", {"name": "simple"})


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


async def test_read_rules_with_stories(default_domain: Domain):

    yaml_file = "data/test_yaml_stories/stories_and_rules.yml"

    steps = await loading.load_data_from_files(
        [yaml_file], default_domain, RegexInterpreter()
    )

    ml_steps = [s for s in steps if not s.is_rule]
    rule_steps = [s for s in steps if s.is_rule]

    # this file contains three rules and three ML stories
    assert len(ml_steps) == 3
    assert len(rule_steps) == 3

    assert rule_steps[0].block_name == "rule 1"
    assert rule_steps[1].block_name == "rule 2"
    assert rule_steps[2].block_name == "rule 3"

    assert ml_steps[0].block_name == "simple_story_without_checkpoint"
    assert ml_steps[1].block_name == "simple_story_with_only_start"
    assert ml_steps[2].block_name == "simple_story_with_only_end"


async def test_read_rules_without_stories(default_domain: Domain):

    yaml_file = "data/test_yaml_stories/rules_without_stories.yml"

    steps = await loading.load_data_from_files(
        [yaml_file], default_domain, RegexInterpreter()
    )

    ml_steps = [s for s in steps if not s.is_rule]
    rule_steps = [s for s in steps if s.is_rule]

    # this file contains three rules and no ML stories
    assert len(ml_steps) == 0
    assert len(rule_steps) == 3

    assert rule_steps[0].block_name == "rule 1"
    assert rule_steps[1].block_name == "rule 2"
    assert rule_steps[2].block_name == "rule 3"

    # inspect the first rule and make sure all events were picked up correctly
    events = rule_steps[0].events

    assert len(events) == 5

    assert events[0] == Form("loop_q_form")
    assert events[1] == SlotSet("requested_slot", "some_slot")
    assert events[2] == ActionExecuted("...")
    assert events[3] == UserUttered(
        "inform",
        {"name": "inform", "confidence": 1.0},
        [{"entity": "some_slot", "value": "bla"}],
    )
    assert events[4] == ActionExecuted("loop_q_form")


async def test_warning_if_intent_not_in_domain(default_domain: Domain):
    stories = """
    stories:
    - story: I am gonna make you explode 💥
      steps:
      # Intent defined in user key.
      - intent: definitely not in domain
    """

    reader = YAMLStoryReader(RegexInterpreter(), default_domain)
    yaml_content = io_utils.read_yaml(stories)

    with pytest.warns(UserWarning):
        reader.read_from_parsed_yaml(yaml_content)


async def test_no_warning_if_intent_in_domain(default_domain: Domain):
    stories = """
    stories:
    - story: I am fine 💥
      steps:
      - intent: greet
    """

    reader = YAMLStoryReader(RegexInterpreter(), default_domain)
    yaml_content = io_utils.read_yaml(stories)

    with pytest.warns(None) as record:
        reader.read_from_parsed_yaml(yaml_content)

    assert len(record) == 0
