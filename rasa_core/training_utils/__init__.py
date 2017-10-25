from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.interpreter import RegexInterpreter
from rasa_core.training_utils.dsl import StoryFileReader, STORY_START, \
    TrainingsDataExtractor
from rasa_core.training_utils.story_graph import StoryGraph


def extract_training_data_from_file(filename,
                                    augmentation_factor=20,
                                    max_history=1,
                                    remove_duplicates=True,
                                    domain=None,
                                    featurizer=None,
                                    interpreter=RegexInterpreter(),
                                    max_number_of_trackers=2000):
    graph = extract_story_graph_from_file(filename, domain, interpreter)
    extractor = TrainingsDataExtractor(graph, domain, featurizer)
    return extractor.extract_trainings_data(remove_duplicates,
                                            augmentation_factor,
                                            max_history,
                                            max_number_of_trackers)


def extract_stories_from_file(filename,
                              domain,
                              interpreter=RegexInterpreter(),
                              max_number_of_trackers=2000):
    graph = extract_story_graph_from_file(filename, domain, interpreter)
    return graph.build_stories(domain,
                               max_number_of_trackers)


def extract_story_graph_from_file(filename, domain,
                                  interpreter=RegexInterpreter()):
    story_steps = StoryFileReader.read_from_file(filename, domain, interpreter)
    return StoryGraph(story_steps)
