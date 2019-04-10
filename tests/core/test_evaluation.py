import os

from rasa.core.server import nlu_model_and_evaluation_files_from_archive
from rasa.core.test import _generate_trackers, collect_story_predictions, test
from rasa.model import add_evaluation_file_to_model

# we need this import to ignore the warning...
# noinspection PyUnresolvedReferences
from rasa.nlu.test import run_evaluation
from tests.core.conftest import (
    DEFAULT_STORIES_FILE,
    E2E_STORY_FILE_UNKNOWN_ENTITY,
    END_TO_END_STORY_FILE,
)


async def test_evaluation_image_creation(tmpdir, default_agent):
    stories_path = os.path.join(tmpdir.strpath, "failed_stories.md")
    img_path = os.path.join(tmpdir.strpath, "story_confmat.pdf")

    await test(
        stories=DEFAULT_STORIES_FILE,
        agent=default_agent,
        out_directory=tmpdir.strpath,
        max_stories=None,
        use_e2e=False,
    )

    assert os.path.isfile(img_path)
    assert os.path.isfile(stories_path)


async def test_action_evaluation_script(tmpdir, default_agent):
    completed_trackers = await _generate_trackers(
        DEFAULT_STORIES_FILE, default_agent, use_e2e=False
    )
    story_evaluation, num_stories = collect_story_predictions(
        completed_trackers, default_agent, use_e2e=False
    )

    assert not story_evaluation.evaluation_store.has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 0
    assert num_stories == 3


async def test_end_to_end_evaluation_script(tmpdir, default_agent):
    completed_trackers = await _generate_trackers(
        END_TO_END_STORY_FILE, default_agent, use_e2e=True
    )

    story_evaluation, num_stories = collect_story_predictions(
        completed_trackers, default_agent, use_e2e=True
    )

    assert not story_evaluation.evaluation_store.has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 0
    assert num_stories == 2


async def test_end_to_end_evaluation_script_unknown_entity(tmpdir, default_agent):
    completed_trackers = await _generate_trackers(
        E2E_STORY_FILE_UNKNOWN_ENTITY, default_agent, use_e2e=True
    )

    story_evaluation, num_stories = collect_story_predictions(
        completed_trackers, default_agent, use_e2e=True
    )

    assert story_evaluation.evaluation_store.has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 1
    assert num_stories == 1


async def test_stack_model_intent_evaluation(
    tmpdir, trained_stack_model, default_nlu_data
):
    with open(default_nlu_data, "r") as f:
        nlu_data = f.read()

    # add evaluation data to model archive
    new_model_path = add_evaluation_file_to_model(
        trained_stack_model, nlu_data, data_format="md"
    )

    nlu_model_path, nlu_files = await nlu_model_and_evaluation_files_from_archive(
        new_model_path, tmpdir
    )

    assert len(nlu_files) == 1
    evaluation = run_evaluation(nlu_files[0], nlu_model_path)

    assert set(evaluation.keys()) == {"intent_evaluation", "entity_evaluation"}
