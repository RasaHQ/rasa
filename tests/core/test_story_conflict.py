from typing import Text, List, Tuple

from rasa.core.domain import Domain
from rasa.core.training.story_conflict import (
    StoryConflict,
    find_story_conflicts,
    _get_previous_event,
)
from rasa.core.training.generator import TrainingDataGenerator, TrackerWithCachedStates
from rasa.validator import Validator
from rasa.importers.rasa import RasaFileImporter
from tests.core.conftest import DEFAULT_STORIES_FILE, DEFAULT_DOMAIN_PATH_WITH_SLOTS


async def _setup_trackers_for_testing(
    domain_path: Text, training_data_file: Text
) -> Tuple[List[TrackerWithCachedStates], Domain]:
    importer = RasaFileImporter(
        domain_path=domain_path, training_data_paths=[training_data_file],
    )
    validator = await Validator.from_importer(importer)

    trackers = TrainingDataGenerator(
        validator.story_graph,
        domain=validator.domain,
        remove_duplicates=False,
        augmentation_factor=0,
    ).generate()

    return trackers, validator.domain


async def test_find_no_conflicts():
    trackers, domain = await _setup_trackers_for_testing(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS, DEFAULT_STORIES_FILE
    )

    # Create a list of `StoryConflict` objects
    conflicts = find_story_conflicts(trackers, domain, 5)

    assert conflicts == []


async def test_find_conflicts_in_short_history():
    trackers, domain = await _setup_trackers_for_testing(
        "data/test_domains/default.yml", "data/test_stories/stories_conflicting_1.md"
    )

    # `max_history = 3` is too small, so a conflict must arise
    conflicts = find_story_conflicts(trackers, domain, 3)
    assert len(conflicts) == 1

    # With `max_history = 4` the conflict should disappear
    conflicts = find_story_conflicts(trackers, domain, 4)
    assert len(conflicts) == 0


async def test_find_conflicts_checkpoints():
    trackers, domain = await _setup_trackers_for_testing(
        "data/test_domains/default.yml", "data/test_stories/stories_conflicting_2.md"
    )

    # Create a list of `StoryConflict` objects
    conflicts = find_story_conflicts(trackers, domain, 5)

    assert len(conflicts) == 1
    assert conflicts[0].conflicting_actions == ["utter_goodbye", "utter_default"]


async def test_find_conflicts_or():
    trackers, domain = await _setup_trackers_for_testing(
        "data/test_domains/default.yml", "data/test_stories/stories_conflicting_3.md"
    )

    # Create a list of `StoryConflict` objects
    conflicts = find_story_conflicts(trackers, domain, 5)

    assert len(conflicts) == 1
    assert conflicts[0].conflicting_actions == ["utter_default", "utter_goodbye"]


async def test_find_conflicts_slots_that_break():
    trackers, domain = await _setup_trackers_for_testing(
        "data/test_domains/default.yml", "data/test_stories/stories_conflicting_4.md"
    )

    # Create a list of `StoryConflict` objects
    conflicts = find_story_conflicts(trackers, domain, 5)

    assert len(conflicts) == 1
    assert conflicts[0].conflicting_actions == ["utter_default", "utter_greet"]


async def test_find_conflicts_slots_that_dont_break():
    trackers, domain = await _setup_trackers_for_testing(
        "data/test_domains/default.yml", "data/test_stories/stories_conflicting_5.md"
    )

    # Create a list of `StoryConflict` objects
    conflicts = find_story_conflicts(trackers, domain, 5)

    assert len(conflicts) == 0


async def test_find_conflicts_multiple_stories():
    trackers, domain = await _setup_trackers_for_testing(
        "data/test_domains/default.yml", "data/test_stories/stories_conflicting_6.md"
    )

    # Create a list of `StoryConflict` objects
    conflicts = find_story_conflicts(trackers, domain, 5)

    assert len(conflicts) == 1
    assert "and 2 other trackers" in str(conflicts[0])


async def test_add_conflicting_action():
    sliced_states = [
        None,
        {},
        {"intent_greet": 1.0, "prev_action_listen": 1.0},
        {"prev_utter_greet": 1.0, "intent_greet": 1.0},
    ]
    conflict = StoryConflict(sliced_states)

    conflict.add_conflicting_action("utter_greet", "xyz")
    conflict.add_conflicting_action("utter_default", "uvw")
    assert conflict.conflicting_actions == ["utter_greet", "utter_default"]


async def test_has_prior_events():
    sliced_states = [
        None,
        {},
        {"intent_greet": 1.0, "prev_action_listen": 1.0},
        {"prev_utter_greet": 1.0, "intent_greet": 1.0},
    ]
    conflict = StoryConflict(sliced_states)
    assert conflict.conflict_has_prior_events


async def test_get_previous_event():
    assert _get_previous_event({"prev_utter_greet": 1.0, "intent_greet": 1.0}) == (
        "action",
        "utter_greet",
    )
    assert _get_previous_event({"intent_greet": 1.0, "prev_utter_greet": 1.0}) == (
        "action",
        "utter_greet",
    )
    assert _get_previous_event({"intent_greet": 1.0, "prev_action_listen": 1.0}) == (
        "intent",
        "greet",
    )


async def test_has_no_prior_events():
    sliced_states = [None]
    conflict = StoryConflict(sliced_states)
    assert not conflict.conflict_has_prior_events
