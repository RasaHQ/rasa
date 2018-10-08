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
from tests.conftest import DEFAULT_STORIES_FILE


def test_evaluation_image_creation(tmpdir, default_agent):
    stories_path = tmpdir.join("failed_stories.md").strpath
    img_path = tmpdir.join("evaluation.png").strpath

    run_story_evaluation(
            resource_name=DEFAULT_STORIES_FILE,
            agent=default_agent,
            out_file_plot=img_path,
            max_stories=None,
            out_file_stories=stories_path
    )

    assert os.path.isfile(img_path)
    assert imghdr.what(img_path) == "png"

    assert os.path.isfile(stories_path)


def test_evaluation_script(tmpdir, default_agent):
    completed_trackers = evaluate._generate_trackers(
            DEFAULT_STORIES_FILE, default_agent)

    golds, predictions, failed_stories, num_stories = collect_story_predictions(
            completed_trackers, default_agent)

    assert len(golds) == 14
    assert len(predictions) == 14
    assert len(failed_stories) == 0
    assert num_stories == 3
