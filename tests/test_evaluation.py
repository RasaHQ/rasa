from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import imghdr
import os

from rasa_core import evaluate
from rasa_core.evaluate import (
    run_story_evaluation,
    collect_story_predictions)
from tests.conftest import DEFAULT_STORIES_FILE, END_TO_END_STORY_FILE


def test_evaluation_image_creation(tmpdir, default_agent):
    stories_path = tmpdir.join("failed_stories.md").strpath
    img_path = tmpdir.join("evaluation.png").strpath

    _ = run_story_evaluation(
            resource_name=DEFAULT_STORIES_FILE,
            agent=default_agent,
            out_file_plot=img_path,
            max_stories=None,
            out_file_stories=stories_path,
            use_e2e=False
    )

    assert os.path.isfile(img_path)
    assert imghdr.what(img_path) == "png"

    assert os.path.isfile(stories_path)


def test_action_evaluation_script(tmpdir, default_agent):
    completed_trackers = evaluate._generate_trackers(
            DEFAULT_STORIES_FILE, default_agent, use_e2e=False)

    evaluation_result, failed_stories, _ = collect_story_predictions(
            completed_trackers, default_agent, use_e2e=False)

    assert not evaluation_result.has_prediction_target_mismatch()
    assert len(failed_stories) == 0


def test_end_to_end_evaluation_script(tmpdir, default_agent):
    completed_trackers = evaluate._generate_trackers(
            END_TO_END_STORY_FILE, default_agent, use_e2e=True)

    evaluation_result, failed_stories, _ = collect_story_predictions(
            completed_trackers, default_agent, use_e2e=True)

    assert not evaluation_result.has_prediction_target_mismatch()
    assert len(failed_stories) == 0
