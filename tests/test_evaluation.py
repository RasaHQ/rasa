from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import imghdr
import os

from rasa_core.evaluate import run_story_evaluation, \
    collect_story_predictions


def test_evaluation_image_creation(tmpdir):
    img_path = os.path.join(tmpdir.strpath, "evaltion.png")
    run_story_evaluation(
            story_file="examples/concerts/data/stories.md",
            policy_model_path="examples/concerts/models/policy/init",
            nlu_model_path=None,
            out_file=img_path,
            max_stories=None
    )

    assert os.path.isfile(img_path)
    assert imghdr.what(img_path) == "png"


def test_evaluation_script():
    actual, preds = collect_story_predictions(
            story_file="examples/concerts/data/stories.md",
            policy_model_path="examples/concerts/models/policy/init",
            nlu_model_path=None,
            max_stories=None,
            shuffle_stories=False
    )
    assert len(actual) == 14
    assert len(preds) == 14
