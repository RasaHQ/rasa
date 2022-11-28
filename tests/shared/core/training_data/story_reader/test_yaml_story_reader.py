import json
import warnings

import numpy as np
import os
import pytest
import sys

from collections import Counter
from pathlib import Path
from typing import Any, Text, List, Dict, Optional
from _pytest.monkeypatch import MonkeyPatch
from unittest.mock import Mock

from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
import rasa.shared.utils.io
from rasa.shared.exceptions import (
    FileNotFoundException,
    YamlSyntaxException,
    YamlException,
)
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core import training
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.utils.tensorflow.model_data_utils import _surface_attributes

from rasa.shared.constants import (
    INTENT_MESSAGE_PREFIX,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
)
from rasa.shared.core.constants import RULE_SNIPPET_ACTION_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data import loading
from rasa.shared.core.events import ActionExecuted, UserUttered, SlotSet, ActiveLoop
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
    DEFAULT_VALUE_TEXT_SLOTS,
)
from rasa.shared.core.training_data.structures import StoryStep, RuleStep
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    ENTITIES,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    FEATURE_TYPE_SENTENCE,
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
    EXTRACTOR,
)
from tests.conftest import filter_expected_warnings


@pytest.fixture()
def rule_steps_without_stories(domain: Domain) -> List[StoryStep]:
    yaml_file = "data/test_yaml_stories/rules_without_stories.yml"

    return loading.load_data_from_files([yaml_file], domain)


def test_can_read_test_story_with_slots(domain: Domain):
    trackers = training.load_data(
        "data/test_yaml_stories/simple_story_with_only_end.yml",
        domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 1

    assert trackers[0].events[-2] == SlotSet(key="name", value="peter")
    assert trackers[0].events[-1] == ActionExecuted("action_listen")


@pytest.mark.parametrize(
    "domain_dict",
    [
        {"slots": {"my_slot": {"type": "text", "mappings": [{"type": "from_text"}]}}},
        {"slots": {"my_slot": {"type": "list", "mappings": [{"type": "from_text"}]}}},
    ],
)
async def test_default_slot_value_if_slots_referenced_by_name_only(domain_dict: Dict):
    story = """
    stories:
    - story: my story
      steps:
      - intent: greet
      - slot_was_set:
        - my_slot
    """

    reader = YAMLStoryReader(Domain.from_dict(domain_dict))
    events = reader.read_from_string(story)[0].events

    assert isinstance(events[-1], SlotSet)
    assert events[-1].value


@pytest.mark.parametrize(
    "domain_dict",
    [
        {
            "slots": {
                "my_slot": {"type": "categorical", "mappings": [{"type": "from_text"}]}
            }
        },
        {"slots": {"my_slot": {"type": "float", "mappings": [{"type": "from_text"}]}}},
    ],
)
async def test_default_slot_value_if_incompatible_slots_referenced_by_name_only(
    domain_dict: Dict,
):
    story = """
    stories:
    - story: my story
      steps:
      - intent: greet
      - slot_was_set:
        - my_slot
    """

    reader = YAMLStoryReader(Domain.from_dict(domain_dict))
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
    with warnings.catch_warnings() as record:
        events = reader.read_from_string(story)[0].events

    if record is not None:
        record = filter_expected_warnings(record)
        assert len(record) == 0

    assert isinstance(events[-1], SlotSet)
    assert events[-1].value is None


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
        {
            "intents": ["greet"],
            "slots": {"my_slot": {"type": "any", "mappings": [{"type": "from_text"}]}},
        }
    )
    reader = YAMLStoryReader(domain)

    with warnings.catch_warnings() as warning:
        events = reader.read_from_string(story)[0].events

    if warning is not None:
        warning = filter_expected_warnings(warning)
        assert len(warning) == 0

    assert isinstance(events[-1], SlotSet)
    assert events[-1].value is None


def test_can_read_test_story_with_entities(domain: Domain):
    trackers = training.load_data(
        "data/test_yaml_stories/story_with_or_and_entities.yml",
        domain,
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


def test_can_read_test_story_with_entities_without_value(domain: Domain):
    trackers = training.load_data(
        "data/test_yaml_stories/story_with_or_and_entities_with_no_value.yml",
        domain,
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
    "file",
    [
        "data/test_yaml_stories/stories.yml",
        "data/test_yaml_stories/rules_without_stories.yml",
    ],
)
async def test_is_yaml_file(file: Text):
    assert YAMLStoryReader.is_stories_file(file) is True


def test_yaml_intent_with_leading_slash_warning(domain: Domain):
    yaml_file = "data/test_wrong_yaml_stories/intent_with_leading_slash.yml"

    with pytest.warns() as record:
        tracker = training.load_data(
            yaml_file,
            domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False,
        )
    record = filter_expected_warnings(record)
    # one for leading slash
    assert len(record) == 1
    assert type(record[0].message) == UserWarning

    assert tracker[0].latest_message == UserUttered(intent={"name": "simple"})


def test_yaml_slot_without_value_is_parsed(domain: Domain):
    yaml_file = "data/test_yaml_stories/story_with_slot_was_set.yml"

    tracker = training.load_data(
        yaml_file,
        domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )

    assert tracker[0].events[-2] == SlotSet(key="name", value=DEFAULT_VALUE_TEXT_SLOTS)


def test_yaml_wrong_yaml_format_warning(domain: Domain):
    yaml_file = "data/test_wrong_yaml_stories/wrong_yaml.yml"

    with pytest.raises(YamlSyntaxException):
        _ = training.load_data(
            yaml_file,
            domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False,
        )


def test_read_rules_with_stories(domain: Domain):

    yaml_file = "data/test_yaml_stories/stories_and_rules.yml"

    steps = loading.load_data_from_files([yaml_file], domain)

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


async def test_warning_if_intent_not_in_domain(domain: Domain):
    stories = """
    stories:
    - story: I am gonna make you explode 💥
      steps:
      # Intent defined in user key.
      - intent: definitely not in domain
    """

    reader = YAMLStoryReader(domain)
    yaml_content = rasa.shared.utils.io.read_yaml(stories)

    with pytest.warns(UserWarning) as record:
        reader.read_from_parsed_yaml(yaml_content)

    # one for missing intent
    assert len(record) == 1


async def test_no_warning_if_intent_in_domain(domain: Domain):
    stories = (
        f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
        f"stories:\n"
        f"- story: I am fine 💥\n"
        f"  steps:\n"
        f"  - intent: greet"
    )

    reader = YAMLStoryReader(domain)
    yaml_content = rasa.shared.utils.io.read_yaml(stories)

    with pytest.warns(None) as record:
        reader.read_from_parsed_yaml(yaml_content)

    assert not len(record)


def test_parsing_of_e2e_stories(domain: Domain):
    yaml_file = "data/test_yaml_stories/stories_hybrid_e2e.yml"
    tracker = training.load_data(
        yaml_file,
        domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )

    assert len(tracker) == 1

    actual = list(tracker[0].events)

    expected = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(intent={"name": "simple"}),
        ActionExecuted("utter_greet"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(
            "I am looking for a Kenyan restaurant",
            {"name": None},
            entities=[{"start": 19, "end": 25, "value": "Kenyan", "entity": "cuisine"}],
        ),
        ActionExecuted("", action_text="good for you"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(intent={"name": "goodbye"}),
        ActionExecuted("utter_goodbye"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("One more thing", {"name": None}),
        ActionExecuted("", action_text="What?"),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]

    assert actual == expected


async def test_active_loop_is_parsed(domain: Domain):
    stories = (
        f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
        f"stories:\n"
        f"- story: name\n"
        f"  steps:\n"
        f"  - intent: greet\n"
        f"  - active_loop: null"
    )

    reader = YAMLStoryReader(domain)
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


def test_read_mixed_training_data_file(domain: Domain):
    training_data_file = "data/test_mixed_yaml_training_data/training_data.yml"

    reader = YAMLStoryReader(domain)
    yaml_content = rasa.shared.utils.io.read_yaml_file(training_data_file)

    with pytest.warns(None) as record:
        reader.read_from_parsed_yaml(yaml_content)
        assert not len(record)


def test_or_statement_with_slot_was_set():
    stories = """
    stories:
    - story: tell name bob or joe
      steps:
      - intent: greet
      - action: utter_greet
      - intent: tell_name
      - or:
        - slot_was_set:
            - name: joe
        - slot_was_set:
            - name: bob
        - slot_was_set:
            - name: null
    """

    reader = YAMLStoryReader()
    yaml_content = rasa.shared.utils.io.read_yaml(stories)

    steps = reader.read_from_parsed_yaml(yaml_content)

    assert len(steps) == 3

    slot = steps[0].events[3]
    assert isinstance(slot, SlotSet)
    assert slot.key == "name"
    assert slot.value == "joe"

    slot = steps[1].events[3]
    assert isinstance(slot, SlotSet)
    assert slot.key == "name"
    assert slot.value == "bob"

    slot = steps[2].events[3]
    assert isinstance(slot, SlotSet)
    assert slot.key == "name"
    assert slot.value is None


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
    reader = YAMLStoryReader()

    with warnings.catch_warnings() as record:
        reader.read_from_file(file)

    if record is not None:
        record = filter_expected_warnings(record)

        if warning:
            assert len(record) == 1
            assert type(record[0].message) == warning
        else:
            assert len(record) == 0


def test_or_statement_story_with_or_slot_was_set(domain: Domain):
    training_trackers = training.load_data(
        "data/test_yaml_stories/story_with_or_slot_was_set.yml",
        domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(training_trackers) == 2
    assert training_trackers[0].events[3] == SlotSet(key="name", value="peter")
    assert training_trackers[1].events[3] == SlotSet(key="name", value="bob")


@pytest.mark.parametrize("is_conversation_test", [True, False])
def test_handles_mixed_steps_for_test_and_e2e_stories(is_conversation_test):
    stories = """
    stories:
    - story: hello world
      steps:
      - user: Hi
      - bot: Hello?
      - user: Well...
        intent: suspicion
    """

    reader = YAMLStoryReader()
    yaml_content = rasa.shared.utils.io.read_yaml(stories)

    steps = reader.read_from_parsed_yaml(yaml_content)

    events = steps[0].events
    assert len(events) == 3
    assert events[0].text == "Hi"
    assert events[1].action_text == "Hello?"
    assert events[2].text == "Well..."


def test_read_from_file_skip_validation(monkeypatch: MonkeyPatch):
    yaml_file = "data/test_wrong_yaml_stories/wrong_yaml.yml"
    reader = YAMLStoryReader()

    monkeypatch.setattr(
        sys.modules["rasa.shared.utils.io"],
        rasa.shared.utils.io.read_yaml.__name__,
        Mock(return_value={}),
    )

    with pytest.raises(YamlException):
        _ = reader.read_from_file(yaml_file, skip_validation=False)

    assert reader.read_from_file(yaml_file, skip_validation=True) == []


@pytest.mark.parametrize(
    "file",
    [
        "data/test_yaml_stories/rules_missing_intent.yml",
        "data/test_yaml_stories/stories_missing_intent.yml",
    ],
)
def test_raises_exception_missing_intent_in_rules(file: Text, domain: Domain):
    reader = YAMLStoryReader(domain)

    with pytest.warns() as warning:
        reader.read_from_file(file)

    warning = filter_expected_warnings(warning)

    assert "Missing intent value" in warning[0].message.args[0]


def test_can_read_test_story(domain: Domain):
    trackers = training.load_data(
        "data/test_yaml_stories/stories.yml",
        domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 7
    # this should be the story simple_story_with_only_end -> show_it_all
    # the generated stories are in a non stable order - therefore we need to
    # do some trickery to find the one we want to test
    tracker = [t for t in trackers if len(t.events) == 5][0]
    assert tracker.events[0] == ActionExecuted("action_listen")
    assert tracker.events[1] == UserUttered(
        intent={INTENT_NAME_KEY: "simple", "confidence": 1.0},
        parse_data={
            "text": "/simple",
            "intent_ranking": [{"confidence": 1.0, INTENT_NAME_KEY: "simple"}],
            "intent": {"confidence": 1.0, INTENT_NAME_KEY: "simple"},
            "entities": [],
        },
    )
    assert tracker.events[2] == ActionExecuted("utter_default")
    assert tracker.events[3] == ActionExecuted("utter_greet")
    assert tracker.events[4] == ActionExecuted("action_listen")


def test_can_read_test_story_with_checkpoint_after_or(domain: Domain):
    trackers = training.load_data(
        "data/test_yaml_stories/stories_checkpoint_after_or.yml",
        domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 2


def test_read_story_file_with_cycles(domain: Domain):
    graph = training.extract_story_graph(
        "data/test_yaml_stories/stories_with_cycle.yml", domain
    )

    assert len(graph.story_steps) == 5

    graph_without_cycles = graph.with_cycles_removed()

    assert graph.cyclic_edge_ids != set()
    # sorting removed_edges converting set converting it to list
    assert graph_without_cycles.cyclic_edge_ids == list()

    assert len(graph.story_steps) == len(graph_without_cycles.story_steps) == 5

    assert len(graph_without_cycles.story_end_checkpoints) == 2


def test_generate_training_data_with_cycles(domain: Domain):
    featurizer = MaxHistoryTrackerFeaturizer(SingleStateFeaturizer(), max_history=4)
    training_trackers = training.load_data(
        "data/test_yaml_stories/stories_with_cycle.yml", domain, augmentation_factor=0
    )

    _, label_ids, _ = featurizer.featurize_trackers(
        training_trackers, domain, precomputations=None
    )

    # how many there are depends on the graph which is not created in a
    # deterministic way but should always be 3 or 4
    assert len(training_trackers) == 3 or len(training_trackers) == 4

    # if we have 4 trackers, there is going to be one example more for label 10
    num_tens = len(training_trackers) - 1
    # if new default actions are added the keys of the actions will be changed

    all_label_ids = [id for ids in label_ids for id in ids]
    assert Counter(all_label_ids) == {0: 6, 15: 3, 14: num_tens, 1: 2, 16: 1}


def test_generate_training_data_with_unused_checkpoints(domain: Domain):
    training_trackers = training.load_data(
        "data/test_yaml_stories/stories_unused_checkpoints.yml", domain
    )
    # there are 3 training stories:
    #   2 with unused end checkpoints -> training_trackers
    #   1 with unused start checkpoints -> ignored
    assert len(training_trackers) == 2


def test_generate_training_data_original_and_augmented_trackers(domain: Domain):
    training_trackers = training.load_data(
        "data/test_yaml_stories/stories_defaultdomain.yml",
        domain,
        augmentation_factor=3,
    )
    # there are three original stories
    # augmentation factor of 3 indicates max of 3*10 augmented stories generated
    # maximum number of stories should be augmented+original = 33
    original_trackers = [
        t
        for t in training_trackers
        if not hasattr(t, "is_augmented") or not t.is_augmented
    ]
    assert len(original_trackers) == 4
    assert len(training_trackers) <= 34


def test_visualize_training_data_graph(tmp_path: Path, domain: Domain):
    graph = training.extract_story_graph(
        "data/test_yaml_stories/stories_with_cycle.yml", domain
    )

    graph = graph.with_cycles_removed()

    out_path = str(tmp_path / "graph.html")

    # this will be the plotted networkx graph
    G = graph.visualize(out_path)

    assert os.path.exists(out_path)

    # we can't check the exact topology - but this should be enough to ensure
    # the visualisation created a sane graph
    assert set(G.nodes()) == set(range(-1, 13)) or set(G.nodes()) == set(range(-1, 14))
    if set(G.nodes()) == set(range(-1, 13)):
        assert len(G.edges()) == 14
    elif set(G.nodes()) == set(range(-1, 14)):
        assert len(G.edges()) == 16


def test_load_multi_file_training_data(domain: Domain):
    featurizer = MaxHistoryTrackerFeaturizer(SingleStateFeaturizer(), max_history=2)
    trackers = training.load_data(
        "data/test_yaml_stories/stories.yml", domain, augmentation_factor=0
    )
    trackers = sorted(trackers, key=lambda t: t.sender_id)

    (tr_as_sts, tr_as_acts) = featurizer.training_states_and_labels(trackers, domain)
    hashed = []
    for sts, acts in zip(tr_as_sts, tr_as_acts):
        hashed.append(json.dumps(sts + acts, sort_keys=True))
    hashed = sorted(hashed, reverse=True)

    data, label_ids, _ = featurizer.featurize_trackers(
        trackers, domain, precomputations=None
    )

    featurizer_mul = MaxHistoryTrackerFeaturizer(SingleStateFeaturizer(), max_history=2)
    trackers_mul = training.load_data(
        "data/test_multifile_yaml_stories", domain, augmentation_factor=0
    )
    trackers_mul = sorted(trackers_mul, key=lambda t: t.sender_id)

    (tr_as_sts_mul, tr_as_acts_mul) = featurizer.training_states_and_labels(
        trackers_mul, domain
    )
    hashed_mul = []
    for sts_mul, acts_mul in zip(tr_as_sts_mul, tr_as_acts_mul):
        hashed_mul.append(json.dumps(sts_mul + acts_mul, sort_keys=True))
    hashed_mul = sorted(hashed_mul, reverse=True)

    data_mul, label_ids_mul, _ = featurizer_mul.featurize_trackers(
        trackers_mul, domain, precomputations=None
    )

    assert hashed == hashed_mul
    # we check for intents, action names and entities -- the features which
    # are included in the story files

    data = _surface_attributes(data)
    data_mul = _surface_attributes(data_mul)

    for attribute in [INTENT, ACTION_NAME, ENTITIES]:
        if attribute not in data or attribute not in data_mul:
            continue
        assert len(data.get(attribute)) == len(data_mul.get(attribute))

        for idx_tracker in range(len(data.get(attribute))):
            for idx_dialogue in range(len(data.get(attribute)[idx_tracker])):
                f1 = data.get(attribute)[idx_tracker][idx_dialogue]
                f2 = data_mul.get(attribute)[idx_tracker][idx_dialogue]
                if f1 is None or f2 is None:
                    assert f1 == f2
                    continue
                for idx_turn in range(len(f1)):
                    f1 = data.get(attribute)[idx_tracker][idx_dialogue][idx_turn]
                    f2 = data_mul.get(attribute)[idx_tracker][idx_dialogue][idx_turn]
                    assert np.all((f1 == f2).data)

    assert np.all(label_ids == label_ids_mul)


def test_yaml_slot_different_types(domain: Domain):
    with pytest.warns(None):
        tracker = training.load_data(
            "data/test_yaml_stories/story_slot_different_types.yml",
            domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False,
        )

    assert tracker[0].events[3] == SlotSet(key="list_slot", value=["value1", "value2"])
    assert tracker[0].events[4] == SlotSet(key="bool_slot", value=True)
    assert tracker[0].events[5] == SlotSet(key="text_slot", value="some_text")


@pytest.mark.parametrize(
    "confidence,entities,expected_confidence,expected_entities,should_warn",
    [
        # easy examples - where entities or intents might be missing
        (None, None, 1.0, [], False),
        ("0.2134345", None, 0.2134345, [], False),
        ("0", None, 0, [], False),
        (
            None,
            json.dumps({"entity1": "entity_value1", "entity2": 2.0}),
            1.0,
            [
                {
                    ENTITY_ATTRIBUTE_TYPE: "entity1",
                    ENTITY_ATTRIBUTE_VALUE: "entity_value1",
                },
                {ENTITY_ATTRIBUTE_TYPE: "entity2", ENTITY_ATTRIBUTE_VALUE: 2.0},
            ],
            False,
        ),
        # malformed confidences
        (
            "-2",
            None,
            1.0,
            [],
            True,
        ),  # no confidence string; some unidentified part left
        ("abc0.2134345", None, 1.0, [], True),  # same
        ("123", None, 1.0, [], True),  # value extracted by > 1
        ("123?", None, 1.0, [], True),  # value extracted by > 1
        ("1.0.", None, 0.0, [], True),  # confidence string extracted but not a float
        # malformed entities
        (None, json.dumps({"entity1": "entity2"}), 1.0, [], True),
        (None, '{"entity1","entity2":2.0}', 1.0, [], True),
        # ... note: if the confidence is None, the following will raise an error!
        (
            "1.0",
            json.dumps(["entity1"]),
            1.0,
            [],
            True,
        ),  # no entity string extracted; some unexpected string left
    ],
)
def test_process_unpacks_attributes_from_single_message_and_fallsback_if_needed(
    confidence: Optional[Text],
    entities: Optional[Text],
    expected_confidence: float,
    expected_entities: Optional[List[Dict[Text, Any]]],
    should_warn: bool,
):
    # dummy intent
    expected_intent = "my-intent"

    # construct text according to pattern
    text = " \t  " + INTENT_MESSAGE_PREFIX + expected_intent
    if confidence is not None:
        text += f"@{confidence}"
    if entities is not None:
        text += entities
    text += " \t "

    # create a message with some dummy attributes and features
    message = Message(
        data={TEXT: text, INTENT: "extracted-from-the-pattern-text-via-nlu"},
        features=[
            Features(
                features=np.zeros((1, 1)),
                feature_type=FEATURE_TYPE_SENTENCE,
                attribute=TEXT,
                origin="nlu-pipeline",
            )
        ],
    )

    # construct domain from expected intent/entities
    domain_entities = [item[ENTITY_ATTRIBUTE_TYPE] for item in expected_entities]
    domain_intents = [expected_intent] if expected_intent is not None else []
    domain = Domain(
        intents=domain_intents,
        entities=domain_entities,
        slots=[],
        responses={},
        action_names=[],
        forms={},
        data={},
    )

    # extract information
    if should_warn:
        with pytest.warns(UserWarning):
            unpacked_message = YAMLStoryReader.unpack_regex_message(message, domain)
    else:
        unpacked_message = YAMLStoryReader.unpack_regex_message(message, domain)

    assert not unpacked_message.features

    assert set(unpacked_message.data.keys()) == {
        TEXT,
        INTENT,
        INTENT_RANKING_KEY,
        ENTITIES,
    }

    assert unpacked_message.data[TEXT] == message.data[TEXT].strip()

    assert set(unpacked_message.data[INTENT].keys()) == {
        INTENT_NAME_KEY,
        PREDICTED_CONFIDENCE_KEY,
    }
    assert unpacked_message.data[INTENT][INTENT_NAME_KEY] == expected_intent
    assert (
        unpacked_message.data[INTENT][PREDICTED_CONFIDENCE_KEY] == expected_confidence
    )

    intent_ranking = unpacked_message.data[INTENT_RANKING_KEY]
    assert len(intent_ranking) == 1
    assert intent_ranking[0] == {
        INTENT_NAME_KEY: expected_intent,
        PREDICTED_CONFIDENCE_KEY: expected_confidence,
    }
    if expected_entities:
        entity_data: List[Dict[Text, Any]] = unpacked_message.data[ENTITIES]
        assert all(
            set(item.keys())
            == {
                ENTITY_ATTRIBUTE_VALUE,
                ENTITY_ATTRIBUTE_TYPE,
                ENTITY_ATTRIBUTE_START,
                ENTITY_ATTRIBUTE_END,
            }
            for item in entity_data
        )
        assert set(
            (item[ENTITY_ATTRIBUTE_TYPE], item[ENTITY_ATTRIBUTE_VALUE])
            for item in expected_entities
        ) == set(
            (item[ENTITY_ATTRIBUTE_TYPE], item[ENTITY_ATTRIBUTE_VALUE])
            for item in entity_data
        )
    else:
        assert unpacked_message.data[ENTITIES] is not None
        assert len(unpacked_message.data[ENTITIES]) == 0


@pytest.mark.parametrize(
    "intent,entities,expected_intent,domain_entities",
    [
        ("wrong_intent", {"entity": 1.0}, "other_intent", ["entity"]),
        ("my_intent", {"wrong_entity": 1.0}, "my_intent", ["other-entity"]),
        ("wrong_intent", {"wrong_entity": 1.0}, "other_intent", ["other-entity"]),
        # Special case: text "my_intent['entity1']" will be interpreted as the intent.
        # This is not caught via the regex at the moment (intent names can include
        # anything except "{" and "@".)
        ("wrong_entity", ["wrong_entity"], "wrong_entity", ["wrong_entity"]),
    ],
)
def test_process_warns_if_intent_or_entities_not_in_domain(
    intent: Text,
    entities: Optional[Text],
    expected_intent: Text,
    domain_entities: List[Text],
):
    # construct text according to pattern
    text = INTENT_MESSAGE_PREFIX + intent  # do not add a confidence value
    if entities is not None:
        text += json.dumps(entities)
    message = Message(data={TEXT: text})

    # construct domain from expected intent/entities
    domain = Domain(
        intents=[expected_intent],
        entities=domain_entities,
        slots=[],
        responses={},
        action_names=[],
        forms={},
        data={},
    )

    # expect a warning
    with pytest.warns(UserWarning):
        unpacked_message = YAMLStoryReader.unpack_regex_message(message, domain)

    if "wrong" not in intent:
        assert unpacked_message.data[INTENT][INTENT_NAME_KEY] == intent
        if "wrong" in entities:
            assert unpacked_message.data[ENTITIES] is not None
            assert len(unpacked_message.data[ENTITIES]) == 0
    else:
        assert unpacked_message == message


async def test_unpack_regex_message_has_correct_entity_start_and_end():
    entity = "name"
    slot_1 = {entity: "Core"}
    text = f"/greet{json.dumps(slot_1)}"

    message = Message(data={TEXT: text})

    domain = Domain(
        intents=["greet"],
        entities=[entity],
        slots=[],
        responses={},
        action_names=[],
        forms={},
        data={},
    )

    message = YAMLStoryReader.unpack_regex_message(
        message, domain, entity_extractor_name="RegexMessageHandler"
    )

    assert message.data == {
        "text": '/greet{"name": "Core"}',
        "intent": {"name": "greet", "confidence": 1.0},
        "intent_ranking": [{"name": "greet", "confidence": 1.0}],
        "entities": [
            {
                "entity": "name",
                "value": "Core",
                "start": 6,
                "end": 22,
                EXTRACTOR: "RegexMessageHandler",
            }
        ],
    }
