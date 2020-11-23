from pathlib import Path
from typing import Text, List, Dict, Optional

import pytest

from rasa.shared.exceptions import FileNotFoundException, YamlSyntaxException
import rasa.shared.utils.io
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.core import training
from rasa.shared.core.constants import RULE_SNIPPET_ACTION_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data import loading
from rasa.shared.core.events import ActionExecuted, UserUttered, SlotSet, ActiveLoop
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
    DEFAULT_VALUE_TEXT_SLOTS,
)
from rasa.shared.core.training_data.structures import StoryStep, RuleStep


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


@pytest.mark.parametrize(
    "domain",
    [
        {"slots": {"my_slot": {"type": "text"}}},
        {"slots": {"my_slot": {"type": "list"}}},
    ],
)
async def test_default_slot_value_if_slots_referenced_by_name_only(domain: Dict):
    story = """
    stories:
    - story: my story
      steps:
      - intent: greet
      - slot_was_set:
        - my_slot
    """

    reader = YAMLStoryReader(Domain.from_dict(domain))
    events = reader.read_from_string(story)[0].events

    assert isinstance(events[-1], SlotSet)
    assert events[-1].value


@pytest.mark.parametrize(
    "domain",
    [
        {"slots": {"my_slot": {"type": "categorical"}}},
        {"slots": {"my_slot": {"type": "float"}}},
    ],
)
async def test_default_slot_value_if_incompatible_slots_referenced_by_name_only(
    domain: Dict,
):
    story = """
    stories:
    - story: my story
      steps:
      - intent: greet
      - slot_was_set:
        - my_slot
    """

    reader = YAMLStoryReader(Domain.from_dict(domain))
    with pytest.warns(UserWarning):
        events = reader.read_from_string(story)[0].events

    assert isinstance(events[-1], SlotSet)
    assert events[-1].value is None


async def test_default_slot_value_if_no_domain():
    story = """
    stories:
    - story: my story
      steps:
      - intent: greet
      - slot_was_set:
        - my_slot
    """

    reader = YAMLStoryReader()
    with pytest.warns(None) as warnings:
        events = reader.read_from_string(story)[0].events

    assert isinstance(events[-1], SlotSet)
    assert events[-1].value is None
    assert not warnings


async def test_default_slot_value_if_unfeaturized_slot():
    story = """
    stories:
    - story: my story
      steps:
      - intent: greet
      - slot_was_set:
        - my_slot
    """
    domain = Domain.from_dict(
        {"intents": ["greet"], "slots": {"my_slot": {"type": "any"}}}
    )
    reader = YAMLStoryReader(domain)
    with pytest.warns(None) as warnings:
        events = reader.read_from_string(story)[0].events

    assert isinstance(events[-1], SlotSet)
    assert events[-1].value is None
    assert not warnings


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


async def test_can_read_test_story_with_entities_without_value(default_domain: Domain):
    trackers = await training.load_data(
        "data/test_yaml_stories/story_with_or_and_entities_with_no_value.yml",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 1

    assert trackers[0].events[-4] == UserUttered(
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
    assert YAMLStoryReader.is_stories_file(file) == is_yaml_file


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

    assert tracker[0].latest_message == UserUttered(intent={"name": "simple"})


async def test_yaml_slot_without_value_is_parsed(default_domain: Domain):
    yaml_file = "data/test_yaml_stories/story_with_slot_was_set.yml"

    tracker = await training.load_data(
        yaml_file,
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )

    assert tracker[0].events[-2] == SlotSet(key="name", value=DEFAULT_VALUE_TEXT_SLOTS)


async def test_yaml_wrong_yaml_format_warning(default_domain: Domain):
    yaml_file = "data/test_wrong_yaml_stories/wrong_yaml.yml"

    with pytest.raises(YamlSyntaxException):
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

    ml_steps = [s for s in steps if not isinstance(s, RuleStep)]
    rule_steps = [s for s in steps if isinstance(s, RuleStep)]

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
    ml_steps = [s for s in rule_steps_without_stories if not isinstance(s, RuleStep)]
    rule_steps = [s for s in rule_steps_without_stories if isinstance(s, RuleStep)]

    # this file contains five rules and no ML stories
    assert len(ml_steps) == 0
    assert len(rule_steps) == 8


def test_rule_with_condition(rule_steps_without_stories: List[StoryStep]):
    rule = rule_steps_without_stories[0]
    assert rule.block_name == "Rule with condition"
    assert rule.events == [
        ActiveLoop("loop_q_form"),
        SlotSet("requested_slot", "some_slot"),
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        UserUttered(
            intent={"name": "inform", "confidence": 1.0},
            entities=[{"entity": "some_slot", "value": "bla"}],
        ),
        ActionExecuted("loop_q_form"),
    ]


def test_rule_without_condition(rule_steps_without_stories: List[StoryStep]):
    rule = rule_steps_without_stories[1]
    assert rule.block_name == "Rule without condition"
    assert rule.events == [
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        UserUttered(intent={"name": "explain", "confidence": 1.0}),
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
        UserUttered(intent={"name": "explain", "confidence": 1.0}),
        ActionExecuted("utter_explain_some_slot"),
    ]


def test_rule_which_hands_over_at_end(rule_steps_without_stories: List[StoryStep]):
    rule = rule_steps_without_stories[3]
    assert rule.block_name == "Rule after which another action should be predicted"
    assert rule.events == [
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        UserUttered(intent={"name": "explain", "confidence": 1.0}),
        ActionExecuted("utter_explain_some_slot"),
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
    ]


def test_conversation_start_rule(rule_steps_without_stories: List[StoryStep]):
    rule = rule_steps_without_stories[4]
    assert rule.block_name == "Rule which only applies to conversation start"
    assert rule.events == [
        UserUttered(intent={"name": "explain", "confidence": 1.0}),
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
    yaml_content = rasa.shared.utils.io.read_yaml(stories)

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
    yaml_content = rasa.shared.utils.io.read_yaml(stories)

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
    yaml_content = rasa.shared.utils.io.read_yaml(stories)

    with pytest.warns(None) as record:
        reader.read_from_parsed_yaml(yaml_content)

    assert not len(record)


def test_is_test_story_file(tmp_path: Path):
    path = str(tmp_path / "test_stories.yml")
    rasa.shared.utils.io.write_yaml({"stories": []}, path)
    assert YAMLStoryReader.is_test_stories_file(path)


def test_is_not_test_story_file_if_it_doesnt_contain_stories(tmp_path: Path):
    path = str(tmp_path / "test_stories.yml")
    rasa.shared.utils.io.write_yaml({"nlu": []}, path)
    assert not YAMLStoryReader.is_test_stories_file(path)


def test_is_not_test_story_file_raises_if_file_does_not_exist(tmp_path: Path):
    path = str(tmp_path / "test_stories.yml")
    with pytest.raises(FileNotFoundException):
        YAMLStoryReader.is_test_stories_file(path)


def test_is_not_test_story_file_without_test_prefix(tmp_path: Path):
    path = str(tmp_path / "stories.yml")
    rasa.shared.utils.io.write_yaml({"stories": []}, path)
    assert not YAMLStoryReader.is_test_stories_file(path)


def test_end_to_end_story_with_shortcut_intent():
    intent = "greet"
    plain_text = f'/{intent}{{"name": "test"}}'
    story = f"""
stories:
- story: my story
  steps:
  - user: |
      {plain_text}
    intent: {intent}
    """

    story_as_yaml = rasa.shared.utils.io.read_yaml(story)

    steps = YAMLStoryReader().read_from_parsed_yaml(story_as_yaml)
    user_uttered = steps[0].events[0]

    assert user_uttered == UserUttered(
        plain_text,
        intent={"name": intent},
        entities=[{"entity": "name", "start": 6, "end": 22, "value": "test"}],
    )


def test_end_to_end_story_with_entities():
    story = """
stories:
- story: my story
  steps:
  - intent: greet
    entities:
    - city: Berlin
      role: from
    """

    story_as_yaml = rasa.shared.utils.io.read_yaml(story)

    steps = YAMLStoryReader().read_from_parsed_yaml(story_as_yaml)
    user_uttered = steps[0].events[0]

    assert user_uttered == UserUttered(
        None,
        intent={"name": "greet"},
        entities=[{"entity": "city", "value": "Berlin", "role": "from"}],
    )


def test_read_mixed_training_data_file(default_domain: Domain):
    training_data_file = "data/test_mixed_yaml_training_data/training_data.yml"

    reader = YAMLStoryReader(default_domain)
    yaml_content = rasa.shared.utils.io.read_yaml_file(training_data_file)

    with pytest.warns(None) as record:
        reader.read_from_parsed_yaml(yaml_content)
        assert not len(record)


def test_or_statement_if_not_training_mode():
    stories = """
    stories:
    - story: hello world
      steps:
      - or:
        - intent: intent1
        - intent: intent2
      - action: some_action
      - intent: intent3
      - action: other_action
    """

    reader = YAMLStoryReader(is_used_for_training=False)
    yaml_content = rasa.shared.utils.io.read_yaml(stories)

    steps = reader.read_from_parsed_yaml(yaml_content)

    assert len(steps) == 1

    assert len(steps[0].events) == 4  # 4 events in total
    assert len(steps[0].start_checkpoints) == 1
    assert steps[0].start_checkpoints[0].name == "STORY_START"
    assert steps[0].end_checkpoints == []

    or_statement = steps[0].events[0]
    assert isinstance(or_statement, list)  # But first one is a list (OR)

    assert or_statement[0].intent["name"] == "intent1"
    assert or_statement[1].intent["name"] == "intent2"


@pytest.mark.parametrize(
    "file,warning",
    [
        ("data/test_yaml_stories/test_base_retrieval_intent_story.yml", None),
        (
            "data/test_yaml_stories/non_test_full_retrieval_intent_story.yml",
            UserWarning,
        ),
    ],
)
async def test_story_with_retrieval_intent_warns(
    file: Text, warning: Optional["Warning"]
):
    reader = YAMLStoryReader(is_used_for_training=False)

    with pytest.warns(warning) as record:
        reader.read_from_file(file)

    assert len(record) == (1 if warning else 0)
