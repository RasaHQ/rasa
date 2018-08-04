from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import imghdr
import os

from rasa_core.evaluate import run_story_evaluation, \
    collect_story_predictions
from tests.conftest import DEFAULT_STORIES_FILE


def test_evaluation_image_creation(tmpdir, default_agent):
    model_path = tmpdir.join("model").strpath
    stories_path = tmpdir.join("failed_stories.md").strpath
    img_path = tmpdir.join("evaluation.png").strpath

    default_agent.persist(model_path)

    run_story_evaluation(
            resource_name=DEFAULT_STORIES_FILE,
            policy_model_path=model_path,
            nlu_model_path=None,
            out_file_plot=img_path,
            max_stories=None,
            out_file_stories=stories_path
    )

    assert os.path.isfile(img_path)
    assert imghdr.what(img_path) == "png"

    assert os.path.isfile(stories_path)


def test_evaluation_script(tmpdir, default_agent):
    model_path = tmpdir.join("model").strpath
    default_agent.persist(model_path)

    actual, preds, failed_stories = collect_story_predictions(
            resource_name=DEFAULT_STORIES_FILE,
            policy_model_path=model_path,
            nlu_model_path=None,
            max_stories=None)
    assert len(actual) == 14
    assert len(preds) == 14
    assert len(failed_stories) == 0
