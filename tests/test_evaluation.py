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
    model_path = os.path.join(tmpdir.strpath, "model")
    default_agent.persist(model_path)
    img_path = os.path.join(tmpdir.strpath, "evaluation.png")
    run_story_evaluation(
            resource_name=DEFAULT_STORIES_FILE,
            policy_model_path=model_path,
            nlu_model_path=None,
            out_file=img_path,
            max_stories=None
    )

    assert os.path.isfile(img_path)
    assert imghdr.what(img_path) == "png"


def test_evaluation_script(tmpdir, default_agent):
    model_path = os.path.join(tmpdir.strpath, "model")
    default_agent.persist(model_path)
    actual, preds = collect_story_predictions(
            resource_name=DEFAULT_STORIES_FILE,
            policy_model_path=model_path,
            nlu_model_path=None,
            max_stories=None,
            shuffle_stories=False
    )
    assert len(actual) == 14
    assert len(preds) == 14
