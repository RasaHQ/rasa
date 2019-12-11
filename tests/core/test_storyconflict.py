from rasa.core.story_conflict import StoryConflict
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
    conflicts = StoryConflict.find_conflicts(trackers, validator.domain, 5)

    assert conflicts == []


async def test_find_conflicts():
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
    conflicts = StoryConflict.find_conflicts(trackers, validator.domain, 1)

    assert conflicts == []
