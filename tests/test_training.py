from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.training_utils import StoryFileReader
from rasa_core.training_utils.visualization import visualize_stories


def test_story_visualization(default_domain):
    story_steps = StoryFileReader.read_from_file(
            "data/dsl_stories/stories.md", default_domain)
    generated_graph = visualize_stories(story_steps)
    assert len(generated_graph.nodes()) == 19
