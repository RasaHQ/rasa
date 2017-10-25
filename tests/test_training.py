from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.interpreter import RegexInterpreter
from rasa_core.train import train_dialogue_model

from rasa_core.training_utils import StoryFileReader
from rasa_core.training_utils.visualization import visualize_stories
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


def test_story_visualization_script():
    from rasa_core.visualize import create_argument_parser
    assert create_argument_parser() is not None


def test_story_visualization(default_domain):
    story_steps = StoryFileReader.read_from_file(
            "data/dsl_stories/stories.md", default_domain,
            interpreter=RegexInterpreter())
    generated_graph = visualize_stories(story_steps)
    assert len(generated_graph.nodes()) == 21


def test_training_script(tmpdir):
    train_dialogue_model(DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE,
                         tmpdir.strpath,
                         use_online_learning=False,
                         nlu_model_path=None,
                         kwargs={})
