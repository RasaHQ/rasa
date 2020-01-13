from rasa.core.training.story_conflict import StoryConflict, find_story_conflicts
from rasa.core.training.generator import TrainingDataGenerator
from rasa.core.validator import Validator
from rasa.importers.rasa import RasaFileImporter
from tests.core.conftest import DEFAULT_STORIES_FILE, DEFAULT_DOMAIN_PATH_WITH_SLOTS


async def test_find_no_conflicts():
    importer = RasaFileImporter(
        domain_path=DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        training_data_paths=[DEFAULT_STORIES_FILE],
    )
    validator = await Validator.from_importer(importer)

    trackers = TrainingDataGenerator(
        validator.story_graph,
        domain=validator.domain,
        remove_duplicates=False,
        augmentation_factor=0,
    ).generate()

    # Create a list of `StoryConflict` objects
    conflicts = find_story_conflicts(trackers, validator.domain, 5)

    assert conflicts == []


async def test_find_conflicts_in_short_history():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=["data/test_stories/stories_conflicting_1.md"],
    )
    validator = await Validator.from_importer(importer)

    trackers = TrainingDataGenerator(
        validator.story_graph,
        domain=validator.domain,
        remove_duplicates=False,
        augmentation_factor=0,
    ).generate()

    # `max_history = 3` is too small, so a conflict must arise
    conflicts = find_story_conflicts(trackers, validator.domain, 3)
    assert len(conflicts) == 1

    # With `max_history = 4` the conflict should disappear
    conflicts = find_story_conflicts(trackers, validator.domain, 4)
    assert len(conflicts) == 0


async def test_find_conflicts_checkpoints():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=["data/test_stories/stories_conflicting_2.md"],
    )
    validator = await Validator.from_importer(importer)

    trackers = TrainingDataGenerator(
        validator.story_graph,
        domain=validator.domain,
        remove_duplicates=False,
        augmentation_factor=0,
    ).generate()

    # Create a list of `StoryConflict` objects
    conflicts = find_story_conflicts(trackers, validator.domain, 5)

    assert len(conflicts) == 1
    assert conflicts[0].conflicting_actions == ["utter_goodbye", "utter_default"]


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
    assert conflict.has_prior_events


async def test_has_no_prior_events():
    sliced_states = [None]
    conflict = StoryConflict(sliced_states)
    assert not conflict.has_prior_events
