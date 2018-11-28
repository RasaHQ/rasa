import os

from rasa_core import evaluate
from rasa_core.evaluate import (
    run_story_evaluation,
    collect_story_predictions)
from tests.conftest import DEFAULT_STORIES_FILE, END_TO_END_STORY_FILE


def test_evaluation_image_creation(tmpdir, default_agent):
    stories_path = os.path.join(tmpdir.strpath, "failed_stories.md")
    img_path = os.path.join(tmpdir.strpath, "story_confmat.pdf")

    run_story_evaluation(
        resource_name=DEFAULT_STORIES_FILE,
        agent=default_agent,
        out_directory=tmpdir.strpath,
        max_stories=None,
        use_e2e=False
    )

    assert os.path.isfile(img_path)
    assert os.path.isfile(stories_path)


def test_action_evaluation_script(tmpdir, default_agent):
    completed_trackers = evaluate._generate_trackers(
        DEFAULT_STORIES_FILE, default_agent, use_e2e=False)
    story_evaluation, num_stories = collect_story_predictions(
        completed_trackers,
        default_agent,
        use_e2e=False)

    assert not story_evaluation.evaluation_store. \
        has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 0
    assert num_stories == 3


def test_end_to_end_evaluation_script(tmpdir, default_agent):
    completed_trackers = evaluate._generate_trackers(
        END_TO_END_STORY_FILE, default_agent, use_e2e=True)

    story_evaluation, num_stories = collect_story_predictions(
        completed_trackers,
        default_agent,
        use_e2e=True)

    assert not story_evaluation.evaluation_store. \
        has_prediction_target_mismatch()
    assert len(story_evaluation.failed_stories) == 0
    assert num_stories == 2
