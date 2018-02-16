from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Text, List

from rasa_core.interpreter import RegexInterpreter
from rasa_core.training.data import DialogueTrainingData
from rasa_core.training.dsl import StoryFileReader
from rasa_core.training.generator import TrainingsDataGenerator
from rasa_core.training.structures import StoryGraph, STORY_END, STORY_START

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.domain import Domain
    from rasa_core.featurizers import Featurizer
    from rasa_core.interpreter import NaturalLanguageInterpreter


def extract_story_graph_from_file(
        filename,  # type: Text
        domain,  # type: Domain
        interpreter=RegexInterpreter()  # type: NaturalLanguageInterpreter
):
    # type: (...) -> StoryGraph

    story_steps = StoryFileReader.read_from_file(filename, domain, interpreter)
    return StoryGraph(story_steps)


def extract_training_data_from_file(
        filename,  # type: Text
        domain,  # type: Domain
        featurizer=None,  # type: Featurizer
        interpreter=RegexInterpreter(),  # type: NaturalLanguageInterpreter
        augmentation_factor=20,  # type: int
        max_history=1,  # type: int
        remove_duplicates=True,
        max_number_of_trackers=2000  # type: int
):
    # type: (...) -> DialogueTrainingData

    graph = extract_story_graph_from_file(filename, domain, interpreter)
    g = TrainingsDataGenerator(graph, domain, featurizer,
                               remove_duplicates,
                               augmentation_factor,
                               max_history,
                               max_number_of_trackers)
    return g.generate()


def extract_trackers_from_file(
        filename,  # type: Text
        domain,  # type: Domain
        featurizer,  # type: Featurizer
        interpreter=RegexInterpreter(),  # type: NaturalLanguageInterpreter
        max_history=1,  # type: int
        max_number_of_trackers=2000  # type: int
):
    # type: (...) -> List[DialogueStateTracker]

    graph = extract_story_graph_from_file(filename, domain, interpreter)
    g = TrainingsDataGenerator(graph, domain, featurizer,
                               use_story_concatenation=False,
                               max_history=max_history,
                               tracker_limit=1000,
                               remove_duplicates=False,
                               max_number_of_trackers=max_number_of_trackers)
    training_data = g.generate()
    return training_data.metadata["trackers"]
