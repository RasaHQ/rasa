from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.interpreter import RegexInterpreter
from rasa_core.train import train_dialogue_model

from rasa_core.training import StoryFileReader
from rasa_core.training.visualization import visualize_stories
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


def test_story_visualization_script():
    from rasa_core.visualize import create_argument_parser
    assert create_argument_parser() is not None


def test_story_visualization(default_domain):
    story_steps = StoryFileReader.read_from_file(
            "data/test_stories/stories.md", default_domain,
            interpreter=RegexInterpreter())
    generated_graph = visualize_stories(story_steps, default_domain,
                                        output_file=None,
                                        max_history=3,
                                        should_merge_nodes=False)
    # TODO I output the graph, and it looks correct
    assert len(generated_graph.nodes()) == 33

    assert len(generated_graph.edges()) == 36


def test_story_visualization_with_merging(default_domain):
    story_steps = StoryFileReader.read_from_file(
            "data/test_stories/stories.md", default_domain,
            interpreter=RegexInterpreter())
    generated_graph = visualize_stories(story_steps, default_domain,
                                        output_file=None,
                                        max_history=3,
                                        should_merge_nodes=True)
    assert 15 < len(generated_graph.nodes()) < 33

    assert 20 < len(generated_graph.edges()) < 33


def test_training_script(tmpdir):
    train_dialogue_model(DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE,
                         tmpdir.strpath,
                         use_online_learning=False,
                         nlu_model_path=None,
                         kwargs={})
    assert True
