from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Text, List, Optional

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.interpreter import NaturalLanguageInterpreter
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.training.structures import StoryGraph


def extract_story_graph(
        resource_name,  # type: Text
        domain,  # type: Domain
        interpreter=None  # type: Optional[NaturalLanguageInterpreter]
):
    # type: (...) -> StoryGraph
    from rasa_core.interpreter import RegexInterpreter
    from rasa_core.training.dsl import StoryFileReader
    from rasa_core.training.structures import StoryGraph

    if not interpreter:
        interpreter = RegexInterpreter()
    story_steps = StoryFileReader.read_from_folder(resource_name,
                                                   domain, interpreter)
    return StoryGraph(story_steps)


def load_data(
        resource_name,  # type: Text
        domain,  # type: Domain
        remove_duplicates=True,  # type: bool
        augmentation_factor=20,  # type: int
        max_number_of_trackers=2000,  # type: int
        tracker_limit=None,  # type: Optional[int]
        use_story_concatenation=True  # type: bool

):
    # type: (...) -> List[DialogueStateTracker]
    from rasa_core.training import extract_story_graph
    from rasa_core.training.generator import TrainingDataGenerator

    if resource_name:
        graph = extract_story_graph(resource_name, domain)

        g = TrainingDataGenerator(graph, domain,
                                  remove_duplicates,
                                  augmentation_factor,
                                  max_number_of_trackers,
                                  tracker_limit,
                                  use_story_concatenation)
        return g.generate()
    else:
        return []


def persist_data(trackers, path):
    # type: (List[DialogueStateTracker]) -> None
    """Dump a list of dialogue trackers in the story format to disk."""

    for t in trackers:
        t.export_stories_to_file(path)
