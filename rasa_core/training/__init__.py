from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.interpreter import RegexInterpreter
from rasa_core.training.dsl import StoryFileReader
from rasa_core.training.generator import TrainingsDataGenerator
from rasa_core.training.structures import StoryGraph, STORY_END, STORY_START


def extract_story_graph_from_file(filename, domain,
                                  interpreter=RegexInterpreter()):
    story_steps = StoryFileReader.read_from_file(filename, domain, interpreter)
    return StoryGraph(story_steps)


def extract_training_data_from_file(filename,
                                    augmentation_factor=20,
                                    max_history=1,
                                    remove_duplicates=True,
                                    domain=None,
                                    featurizer=None,
                                    interpreter=RegexInterpreter(),
                                    max_number_of_trackers=2000):
    graph = extract_story_graph_from_file(filename, domain, interpreter)
    g = TrainingsDataGenerator(graph, domain, featurizer,
                               remove_duplicates,
                               augmentation_factor,
                               max_history,
                               max_number_of_trackers)
    return g.generate()


def extract_stories_from_file(filename,
                              domain,
                              interpreter=RegexInterpreter(),
                              max_number_of_trackers=2000):
    graph = extract_story_graph_from_file(filename, domain, interpreter)
    return graph.build_stories(domain,
                               max_number_of_trackers)
