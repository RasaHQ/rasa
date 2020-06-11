from rasa.core import training
from rasa.core.events import ActionExecuted, UserUttered, SlotSet


async def test_can_read_test_simple_story(default_domain):
    trackers = await training.load_data(
        "data/test_yaml_stories/stories_1.yml",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 1

    assert trackers[0].events[0] == ActionExecuted("action_listen")
    assert trackers[0].events[1] == UserUttered(
        "simple",
        intent={"name": "simple", "confidence": 1.0},
        parse_data={
            "text": "/simple",
            "intent_ranking": [{"confidence": 1.0, "name": "simple"}],
            "intent": {"confidence": 1.0, "name": "simple"},
            "entities": [],
        },
    )
    assert trackers[0].events[2] == ActionExecuted("utter_default")
    assert trackers[0].events[3] == ActionExecuted("utter_greet")
    assert trackers[0].events[4] == ActionExecuted("action_listen")


async def test_can_read_test_story_with_slots(default_domain):
    trackers = await training.load_data(
        "data/test_yaml_stories/stories_2.yml",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 1

    assert trackers[0].events[-2] == SlotSet(key="name", value="peter")
    assert trackers[0].events[-1] == ActionExecuted("action_listen")


async def test_can_read_test_story_with_entities_slot_autofill(default_domain):
    trackers = await training.load_data(
        "data/test_yaml_stories/stories_3.yml",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 1

    assert trackers[0].events[-4] == UserUttered(
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
    assert trackers[0].events[-3] == SlotSet(key="name", value="peter")
    assert trackers[0].events[-2] == ActionExecuted("utter_greet")
    assert trackers[0].events[-1] == ActionExecuted("action_listen")


async def test_can_read_all_stories_from_folder(default_domain):
    trackers = await training.load_data(
        "data/test_yaml_stories",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 3
