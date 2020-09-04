from pathlib import Path
from typing import Text, List

import pytest

import rasa.utils.io
from rasa.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.core import training
from rasa.core.actions.action import RULE_SNIPPET_ACTION_NAME
from rasa.core.domain import Domain
from rasa.core.training import loading
from rasa.core.events import ActionExecuted, UserUttered, SlotSet, ActiveLoop
from rasa.core.training.story_reader.yaml_story_reader import YAMLStoryReader
from rasa.core.training.structures import StoryStep


@pytest.fixture()
async def rule_steps_without_stories(default_domain: Domain) -> List[StoryStep]:
    yaml_file = "data/test_yaml_stories/rules_without_stories.yml"

    return await loading.load_data_from_files([yaml_file], default_domain)


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

    with pytest.warns(UserWarning) as record:
        tracker = await training.load_data(
            yaml_file,
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False,
        )

    # one for leading slash
    assert len(record) == 1

    assert tracker[0].latest_message == UserUttered("simple", {"name": "simple"})


async def test_yaml_slot_without_value_is_parsed(default_domain: Domain):
    yaml_file = "data/test_yaml_stories/story_with_slot_was_set.yml"

    tracker = await training.load_data(
        yaml_file,
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )

    assert tracker[0].events[-2] == SlotSet(key="name", value=None)


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

    steps = await loading.load_data_from_files([yaml_file], default_domain)

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


def test_read_rules_without_stories(rule_steps_without_stories: List[StoryStep]):
    ml_steps = [s for s in rule_steps_without_stories if not s.is_rule]
    rule_steps = [s for s in rule_steps_without_stories if s.is_rule]

    # this file contains five rules and no ML stories
    assert len(ml_steps) == 0
    assert len(rule_steps) == 5


def test_rule_with_condition(rule_steps_without_stories: List[StoryStep]):
    rule = rule_steps_without_stories[0]
    assert rule.block_name == "Rule with condition"
    assert rule.events == [
        ActiveLoop("loop_q_form"),
        SlotSet("requested_slot", "some_slot"),
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        UserUttered(
            "inform",
            {"name": "inform", "confidence": 1.0},
            [{"entity": "some_slot", "value": "bla"}],
        ),
        ActionExecuted("loop_q_form"),
    ]


def test_rule_without_condition(rule_steps_without_stories: List[StoryStep]):
    rule = rule_steps_without_stories[1]
    assert rule.block_name == "Rule without condition"
    assert rule.events == [
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        UserUttered("explain", {"name": "explain", "confidence": 1.0}, []),
        ActionExecuted("utter_explain_some_slot"),
        ActionExecuted("loop_q_form"),
        ActiveLoop("loop_q_form"),
    ]


def test_rule_with_explicit_wait_for_user_message(
    rule_steps_without_stories: List[StoryStep],
):
    rule = rule_steps_without_stories[2]
    assert rule.block_name == "Rule which explicitly waits for user input when finished"
    assert rule.events == [
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        UserUttered("explain", {"name": "explain", "confidence": 1.0}, []),
        ActionExecuted("utter_explain_some_slot"),
    ]


def test_rule_which_hands_over_at_end(rule_steps_without_stories: List[StoryStep]):
    rule = rule_steps_without_stories[3]
    assert rule.block_name == "Rule after which another action should be predicted"
    assert rule.events == [
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        UserUttered("explain", {"name": "explain", "confidence": 1.0}, []),
        ActionExecuted("utter_explain_some_slot"),
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
    ]


def test_conversation_start_rule(rule_steps_without_stories: List[StoryStep]):
    rule = rule_steps_without_stories[4]
    assert rule.block_name == "Rule which only applies to conversation start"
    assert rule.events == [
        UserUttered("explain", {"name": "explain", "confidence": 1.0}, []),
        ActionExecuted("utter_explain_some_slot"),
    ]


async def test_warning_if_intent_not_in_domain(default_domain: Domain):
    stories = """
    stories:
    - story: I am gonna make you explode 💥
      steps:
      # Intent defined in user key.
      - intent: definitely not in domain
    """

    reader = YAMLStoryReader(default_domain)
    yaml_content = rasa.utils.io.read_yaml(stories)

    with pytest.warns(UserWarning) as record:
        reader.read_from_parsed_yaml(yaml_content)

    # one for missing intent
    assert len(record) == 1


async def test_no_warning_if_intent_in_domain(default_domain: Domain):
    stories = (
        f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
        f"stories:\n"
        f"- story: I am fine 💥\n"
        f"  steps:\n"
        f"  - intent: greet"
    )

    reader = YAMLStoryReader(default_domain)
    yaml_content = rasa.utils.io.read_yaml(stories)

    with pytest.warns(None) as record:
        reader.read_from_parsed_yaml(yaml_content)

    assert not len(record)


async def test_active_loop_is_parsed(default_domain: Domain):
    stories = (
        f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
        f"stories:\n"
        f"- story: name\n"
        f"  steps:\n"
        f"  - intent: greet\n"
        f"  - active_loop: null"
    )

    reader = YAMLStoryReader(default_domain)
    yaml_content = rasa.utils.io.read_yaml(stories)

    with pytest.warns(None) as record:
        reader.read_from_parsed_yaml(yaml_content)

    assert not len(record)


def test_is_test_story_file(tmp_path: Path):
    path = str(tmp_path / "test_stories.yml")
    rasa.utils.io.write_yaml({"stories": []}, path)
    assert YAMLStoryReader.is_yaml_test_stories_file(path)


def test_is_not_test_story_file_if_it_doesnt_contain_stories(tmp_path: Path):
    path = str(tmp_path / "test_stories.yml")
    rasa.utils.io.write_yaml({"nlu": []}, path)
    assert not YAMLStoryReader.is_yaml_test_stories_file(path)


def test_is_not_test_story_file_if_empty(tmp_path: Path):
    path = str(tmp_path / "test_stories.yml")
    assert not YAMLStoryReader.is_yaml_test_stories_file(path)


def test_is_not_test_story_file_without_test_prefix(tmp_path: Path):
    path = str(tmp_path / "stories.yml")
    rasa.utils.io.write_yaml({"stories": []}, path)
    assert not YAMLStoryReader.is_yaml_test_stories_file(path)
