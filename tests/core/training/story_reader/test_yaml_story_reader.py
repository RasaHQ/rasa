import pytest

from rasa.core import training
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, UserUttered, SlotSet
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


async def test_is_yaml_file():
    valid_yaml_file = "data/test_yaml_stories/stories.yml"
    valid_markdown_file = "data/test_stories/stories.md"

    assert YAMLStoryReader.is_yaml_story_file(valid_yaml_file)
    assert not YAMLStoryReader.is_yaml_story_file(valid_markdown_file)


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
        await training.load_data(
            yaml_file,
            default_domain,
            use_story_concatenation=False,
            tracker_limit=1000,
            remove_duplicates=False,
        )


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


async def test_parsing_of_e2e_stories(default_domain: Domain):
    yaml_file = "data/test_yaml_stories/stories_hybrid_e2e.yml"
    tracker = await training.load_data(
        yaml_file,
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )

    assert len(tracker) == 1

    assert list(tracker[0].events) == [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("simple", {"name": "simple"}),
        ActionExecuted("utter_greet"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(
            "I am looking for a Kenyan restaurant",
            {"name": None},
            entities=[{"start": 19, "end": 25, "value": "Kenyan", "entity": "cuisine"}],
        ),
        ActionExecuted("good for you", e2e_text="good for you"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("goodbye", {"name": "goodbye"}),
        ActionExecuted("utter_goodbye"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("One more thing", {"name": None}),
        ActionExecuted("What?", e2e_text="good for you"),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]
