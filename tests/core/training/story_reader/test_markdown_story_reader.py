from pathlib import Path
from typing import Dict, Text

import pytest

import rasa.utils.io
from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.events import (
    UserUttered,
    ActionExecuted,
    ActionExecutionRejected,
    ActiveLoop,
    FormValidation,
    SlotSet,
    LegacyForm,
)
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training import loading
from rasa.core.training.story_reader.markdown_story_reader import MarkdownStoryReader
from rasa.core.training.structures import Story


async def test_persist_and_read_test_story_graph(
    tmp_path: Path, default_domain: Domain
):
    graph = await training.extract_story_graph(
        "data/test_stories/stories.md", default_domain
    )
    out_path = tmp_path / "persisted_story.md"
    rasa.utils.io.write_text_file(graph.as_story_string(), str(out_path))

    recovered_trackers = await training.load_data(
        str(out_path),
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    existing_trackers = await training.load_data(
        "data/test_stories/stories.md",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )

    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)


async def test_persist_and_read_test_story(tmp_path: Path, default_domain: Domain):
    graph = await training.extract_story_graph(
        "data/test_stories/stories.md", default_domain
    )
    out_path = tmp_path / "persisted_story.md"
    Story(graph.story_steps).dump_to_file(str(out_path))

    recovered_trackers = await training.load_data(
        str(out_path),
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    existing_trackers = await training.load_data(
        "data/test_stories/stories.md",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)


async def test_persist_legacy_form_story():
    domain = Domain.load("data/test_domains/form.yml")

    tracker = DialogueStateTracker("", domain.slots)

    story = (
        "* greet\n"
        "    - utter_greet\n"
        "* start_form\n"
        "    - some_form\n"
        '    - form{"name": "some_form"}\n'
        "* default\n"
        "    - utter_default\n"
        "    - some_form\n"
        "* stop\n"
        "    - utter_ask_continue\n"
        "* affirm\n"
        "    - some_form\n"
        "* stop\n"
        "    - utter_ask_continue\n"
        "* inform\n"
        "    - some_form\n"
        '    - form{"name": null}\n'
        "* goodbye\n"
        "    - utter_goodbye\n"
    )

    # simulate talking to the form
    events = [
        UserUttered(intent={"name": "greet"}),
        ActionExecuted("utter_greet"),
        ActionExecuted("action_listen"),
        # start the form
        UserUttered(intent={"name": "start_form"}),
        ActionExecuted("some_form"),
        ActiveLoop("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "default"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_default"),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "stop"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_ask_continue"),
        ActionExecuted("action_listen"),
        # out of form input but continue with the form
        UserUttered(intent={"name": "affirm"}),
        FormValidation(False),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "stop"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_ask_continue"),
        ActionExecuted("action_listen"),
        # form input
        UserUttered(intent={"name": "inform"}),
        FormValidation(True),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        ActiveLoop(None),
        UserUttered(intent={"name": "goodbye"}),
        ActionExecuted("utter_goodbye"),
        ActionExecuted("action_listen"),
    ]
    [tracker.update(e) for e in events]

    story = story.replace(f"- {LegacyForm.type_name}", f"- {ActiveLoop.type_name}")

    assert story in tracker.export_stories()


async def test_persist_form_story():
    domain = Domain.load("data/test_domains/form.yml")

    tracker = DialogueStateTracker("", domain.slots)

    story = (
        "* greet\n"
        "    - utter_greet\n"
        "* start_form\n"
        "    - some_form\n"
        '    - active_loop{"name": "some_form"}\n'
        "* default\n"
        "    - utter_default\n"
        "    - some_form\n"
        "* stop\n"
        "    - utter_ask_continue\n"
        "* affirm\n"
        "    - some_form\n"
        "* stop\n"
        "    - utter_ask_continue\n"
        "* inform\n"
        "    - some_form\n"
        '    - active_loop{"name": null}\n'
        "* goodbye\n"
        "    - utter_goodbye\n"
    )

    # simulate talking to the form
    events = [
        UserUttered(intent={"name": "greet"}),
        ActionExecuted("utter_greet"),
        ActionExecuted("action_listen"),
        # start the form
        UserUttered(intent={"name": "start_form"}),
        ActionExecuted("some_form"),
        ActiveLoop("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "default"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_default"),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "stop"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_ask_continue"),
        ActionExecuted("action_listen"),
        # out of form input but continue with the form
        UserUttered(intent={"name": "affirm"}),
        FormValidation(False),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "stop"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_ask_continue"),
        ActionExecuted("action_listen"),
        # form input
        UserUttered(intent={"name": "inform"}),
        FormValidation(True),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        ActiveLoop(None),
        UserUttered(intent={"name": "goodbye"}),
        ActionExecuted("utter_goodbye"),
        ActionExecuted("action_listen"),
    ]
    [tracker.update(e) for e in events]

    assert story in tracker.export_stories()


async def test_read_stories_with_multiline_comments(tmpdir, default_domain: Domain):
    reader = MarkdownStoryReader(default_domain)

    story_steps = await reader.read_from_file(
        "data/test_stories/stories_with_multiline_comments.md"
    )

    assert len(story_steps) == 4
    assert story_steps[0].block_name == "happy path"
    assert len(story_steps[0].events) == 4
    assert story_steps[1].block_name == "sad path 1"
    assert len(story_steps[1].events) == 7
    assert story_steps[2].block_name == "sad path 2"
    assert len(story_steps[2].events) == 7
    assert story_steps[3].block_name == "say goodbye"
    assert len(story_steps[3].events) == 2


async def test_read_stories_with_rules(default_domain: Domain):
    story_steps = await loading.load_data_from_files(
        ["data/test_stories/stories_with_rules.md"], default_domain
    )

    # this file contains three rules and two ML stories
    assert len(story_steps) == 5

    ml_steps = [s for s in story_steps if not s.is_rule]
    rule_steps = [s for s in story_steps if s.is_rule]

    assert len(ml_steps) == 2
    assert len(rule_steps) == 3

    assert story_steps[0].block_name == "rule 1"
    assert story_steps[1].block_name == "rule 2"
    assert story_steps[2].block_name == "ML story 1"
    assert story_steps[3].block_name == "rule 3"
    assert story_steps[4].block_name == "ML story 2"


async def test_read_rules_without_stories(default_domain: Domain):
    story_steps = await loading.load_data_from_files(
        ["data/test_stories/rules_without_stories.md"], default_domain,
    )

    # this file contains three rules and two ML stories
    assert len(story_steps) == 3

    ml_steps = [s for s in story_steps if not s.is_rule]
    rule_steps = [s for s in story_steps if s.is_rule]

    assert len(ml_steps) == 0
    assert len(rule_steps) == 3

    assert rule_steps[0].block_name == "rule 1"
    assert rule_steps[1].block_name == "rule 2"
    assert rule_steps[2].block_name == "rule 3"

    # inspect the first rule and make sure all events were picked up correctly
    events = rule_steps[0].events

    assert len(events) == 5

    assert events[0] == ActiveLoop("loop_q_form")
    assert events[1] == SlotSet("requested_slot", "some_slot")
    assert events[2] == ActionExecuted("...")
    assert events[3] == UserUttered(
        'inform{"some_slot":"bla"}',
        {"name": "inform", "confidence": 1.0},
        [{"entity": "some_slot", "start": 6, "end": 25, "value": "bla"}],
    )
    assert events[4] == ActionExecuted("loop_q_form")


@pytest.mark.parametrize(
    "line, expected",
    [
        (" greet: hi", {"intent": "greet", "text": "hi"}),
        (" greet: /greet", {"intent": "greet", "text": "/greet", "entities": [],},),
        (
            'greet: /greet{"test": "test"}',
            {
                "intent": "greet",
                "entities": [
                    {"entity": "test", "start": 6, "end": 22, "value": "test"}
                ],
                "text": '/greet{"test": "test"}',
            },
        ),
        (
            'greet{"test": "test"}: /greet{"test": "test"}',
            {
                "intent": "greet",
                "entities": [
                    {"entity": "test", "start": 6, "end": 22, "value": "test"}
                ],
                "text": '/greet{"test": "test"}',
            },
        ),
        (
            "mood_great: [great](feeling)",
            {
                "intent": "mood_great",
                "entities": [
                    {"start": 0, "end": 5, "value": "great", "entity": "feeling"}
                ],
                "text": "great",
            },
        ),
        (
            'form: greet{"test": "test"}: /greet{"test": "test"}',
            {
                "intent": "greet",
                "entities": [
                    {"end": 22, "entity": "test", "start": 6, "value": "test"}
                ],
                "text": '/greet{"test": "test"}',
            },
        ),
    ],
)
def test_e2e_parsing(line: Text, expected: Dict):
    actual = MarkdownStoryReader.parse_e2e_message(line)

    assert actual.as_dict() == expected


@pytest.mark.parametrize(
    "parse_data, expected_story_string",
    [
        (
            {
                "text": "/simple",
                "parse_data": {
                    "intent": {"confidence": 1.0, "name": "simple"},
                    "entities": [
                        {"start": 0, "end": 5, "value": "great", "entity": "feeling"}
                    ],
                },
            },
            "simple: /simple",
        ),
        (
            {
                "text": "great",
                "parse_data": {
                    "intent": {"confidence": 1.0, "name": "simple"},
                    "entities": [
                        {"start": 0, "end": 5, "value": "great", "entity": "feeling"}
                    ],
                },
            },
            "simple: [great](feeling)",
        ),
        (
            {
                "text": "great",
                "parse_data": {
                    "intent": {"confidence": 1.0, "name": "simple"},
                    "entities": [],
                },
            },
            "simple: great",
        ),
    ],
)
def test_user_uttered_to_e2e(parse_data: Dict, expected_story_string: Text):
    event = UserUttered.from_story_string("user", parse_data)[0]

    assert isinstance(event, UserUttered)
    assert event.as_story_string(e2e=True) == expected_story_string


@pytest.mark.parametrize("line", [" greet{: hi"])
def test_invalid_end_to_end_format(line: Text):
    reader = MarkdownStoryReader()

    with pytest.raises(ValueError):
        # noinspection PyProtectedMember
        _ = reader.parse_e2e_message(line)
