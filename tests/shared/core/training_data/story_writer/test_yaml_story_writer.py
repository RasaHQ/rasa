from pathlib import Path
import textwrap
from typing import Text
from collections import OrderedDict
import pytest
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION

from rasa.shared.core.constants import (
    ACTION_SESSION_START_NAME,
    ACTION_LISTEN_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    DefinePrevUserUtteredFeaturization,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.core.training_data.structures import STORY_START


@pytest.mark.parametrize(
    "input_yaml_file",
    [
        "data/test_yaml_stories/stories.yml",
        "data/test_yaml_stories/stories_defaultdomain.yml",
    ],
)
async def test_simple_story(tmpdir: Path, domain: Domain, input_yaml_file: Text):
    original_yaml_reader = YAMLStoryReader(domain, None)
    original_yaml_story_steps = original_yaml_reader.read_from_file(input_yaml_file)

    target_story_filename = tmpdir / "test.yml"
    writer = YAMLStoryWriter()
    writer.dump(target_story_filename, original_yaml_story_steps)

    processed_yaml_reader = YAMLStoryReader(domain, None)
    processed_yaml_story_steps = processed_yaml_reader.read_from_file(
        target_story_filename
    )

    assert len(processed_yaml_story_steps) == len(original_yaml_story_steps)
    for processed_step, original_step in zip(
        processed_yaml_story_steps, original_yaml_story_steps
    ):
        assert len(processed_step.events) == len(original_step.events)


async def test_story_start_checkpoint_is_skipped(domain: Domain):
    input_yaml_file = "data/test_yaml_stories/stories.yml"

    original_yaml_reader = YAMLStoryReader(domain, None)
    original_yaml_story_steps = original_yaml_reader.read_from_file(input_yaml_file)

    yaml_text = YAMLStoryWriter().dumps(original_yaml_story_steps)

    assert STORY_START not in yaml_text


async def test_forms_are_converted(domain: Domain):
    original_yaml_reader = YAMLStoryReader(domain, None)
    original_yaml_story_steps = original_yaml_reader.read_from_file(
        "data/test_yaml_stories/stories_form.yml"
    )

    assert YAMLStoryWriter.stories_contain_loops(original_yaml_story_steps)

    writer = YAMLStoryWriter()

    with pytest.warns(None) as record:
        writer.dumps(original_yaml_story_steps)

    assert len(record) == 0


def test_yaml_writer_dumps_user_messages():
    events = [UserUttered("Hello", {"name": "greet"}), ActionExecuted("utter_greet")]
    tracker = DialogueStateTracker.from_events("default", events)
    dump = YAMLStoryWriter().dumps(tracker.as_story().story_steps, is_test_story=True)

    assert (
        dump.strip()
        == textwrap.dedent(
            f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        stories:
        - story: default
          steps:
          - intent: greet
            user: |-
              Hello
          - action: utter_greet

    """
        ).strip()
    )


def test_yaml_writer_doesnt_dump_action_unlikely_intent():
    events = [
        UserUttered("Hello", {"name": "greet"}),
        ActionExecuted("utter_hello"),
        ActionExecuted(ACTION_UNLIKELY_INTENT_NAME, metadata={"key1": "value1"}),
        ActionExecuted("utter_bye"),
    ]
    tracker = DialogueStateTracker.from_events("default", events)
    dump = YAMLStoryWriter().dumps(tracker.as_story().story_steps, is_test_story=True)

    assert (
        dump.strip()
        == textwrap.dedent(
            f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    stories:
    - story: default
      steps:
      - intent: greet
        user: |-
          Hello
      - action: utter_hello
      - action: utter_bye

"""
        ).strip()
    )


def test_yaml_writer_avoids_dumping_not_existing_user_messages():
    events = [UserUttered("greet", {"name": "greet"}), ActionExecuted("utter_greet")]
    tracker = DialogueStateTracker.from_events("default", events)
    dump = YAMLStoryWriter().dumps(tracker.as_story().story_steps)

    assert (
        dump.strip()
        == textwrap.dedent(
            f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        stories:
        - story: default
          steps:
          - intent: greet
          - action: utter_greet

    """
        ).strip()
    )


@pytest.mark.parametrize(
    "input_yaml_file", ["data/test_yaml_stories/rules_with_stories_sorted.yaml"]
)
def test_yaml_writer_dumps_rules(input_yaml_file: Text, tmpdir: Path, domain: Domain):
    original_yaml_reader = YAMLStoryReader(domain, None)
    original_yaml_story_steps = original_yaml_reader.read_from_file(input_yaml_file)

    dump = YAMLStoryWriter().dumps(original_yaml_story_steps)
    # remove the version string
    dump = "\n".join(dump.split("\n")[1:])

    with open(input_yaml_file) as original_file:
        assert dump == original_file.read()


async def test_action_start_action_listen_are_not_dumped():
    events = [
        ActionExecuted(ACTION_SESSION_START_NAME),
        UserUttered("Hello", {"name": "greet"}),
        ActionExecuted("utter_greet"),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]
    tracker = DialogueStateTracker.from_events("default", events)
    dump = YAMLStoryWriter().dumps(tracker.as_story().story_steps)

    assert ACTION_SESSION_START_NAME not in dump
    assert ACTION_LISTEN_NAME not in dump


def test_yaml_writer_stories_to_yaml(domain: Domain):
    reader = YAMLStoryReader(domain, None)
    writer = YAMLStoryWriter()
    steps = reader.read_from_file(
        "data/test_yaml_stories/simple_story_with_only_end.yml"
    )

    result = writer.stories_to_yaml(steps)
    assert isinstance(result, OrderedDict)
    assert "stories" in result
    assert len(result["stories"]) == 1


def test_yaml_writer_stories_to_yaml_with_null_entities(domain: Domain):
    writer = YAMLStoryWriter()
    stories = textwrap.dedent(
        """
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    stories:
    - story: happy path
      steps:
      - intent: test_intent
        entities:
        - test_entity: null
        - test_entity2: false
    """
    )

    stories_yaml = YAMLStoryReader().read_from_string(stories)
    result = writer.stories_to_yaml(stories_yaml)
    assert isinstance(result, OrderedDict)
    assert "stories" in result
    assert len(result["stories"]) == 1
    entities = result["stories"][0]["steps"][0]["entities"]
    assert entities[0] == "test_entity"
    assert entities[1] == OrderedDict({"test_entity2": False})


def test_writing_end_to_end_stories(domain: Domain):
    story_name = "test_writing_end_to_end_stories"
    events = [
        # Training story story with intent and action labels
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(intent={"name": "greet"}),
        ActionExecuted("utter_greet"),
        ActionExecuted(ACTION_LISTEN_NAME),
        # Prediction story story with intent and action labels
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(text="Hi", intent={"name": "greet"}),
        DefinePrevUserUtteredFeaturization(use_text_for_featurization=False),
        ActionExecuted("utter_greet"),
        ActionExecuted(ACTION_LISTEN_NAME),
        # End-To-End Training Story
        UserUttered(text="Hi"),
        ActionExecuted(action_text="Hi, I'm a bot."),
        ActionExecuted(ACTION_LISTEN_NAME),
        # End-To-End Prediction Story
        UserUttered("Hi", intent={"name": "greet"}),
        DefinePrevUserUtteredFeaturization(use_text_for_featurization=True),
        ActionExecuted(action_text="Hi, I'm a bot."),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]

    tracker = DialogueStateTracker.from_events(story_name, events)
    dump = YAMLStoryWriter().dumps(tracker.as_story().story_steps)

    assert (
        dump.strip()
        == textwrap.dedent(
            f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        stories:
        - story: {story_name}
          steps:
          - intent: greet
          - action: utter_greet
          - intent: greet
          - action: utter_greet
          - user: |-
              Hi
          - bot: Hi, I'm a bot.
          - user: |-
              Hi
          - bot: Hi, I'm a bot.
    """
        ).strip()
    )


def test_reading_and_writing_end_to_end_stories_in_test_mode(domain: Domain):
    story_name = "test_writing_end_to_end_stories_in_test_mode"

    conversation_tests = f"""
stories:
- story: {story_name}
  steps:
  - intent: greet
    user: Hi
  - action: utter_greet
  - intent: greet
    user: |
      [Hi](test)
  - action: utter_greet
  - user: Hi
  - bot: Hi, I'm a bot.
  - user: |
      [Hi](test)
  - bot: Hi, I'm a bot.
    """

    end_to_end_tests = YAMLStoryReader().read_from_string(conversation_tests)
    dump = YAMLStoryWriter().dumps(end_to_end_tests, is_test_story=True)

    assert (
        dump.strip()
        == textwrap.dedent(
            f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        stories:
        - story: {story_name}
          steps:
          - intent: greet
            user: |-
              Hi
          - action: utter_greet
          - intent: greet
            user: |-
              [Hi](test)
          - action: utter_greet
          - user: |-
              Hi
          - bot: Hi, I'm a bot.
          - user: |-
              [Hi](test)
          - bot: Hi, I'm a bot.
    """
        ).strip()
    )
