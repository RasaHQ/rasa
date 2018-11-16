import typing
from typing import Text, List, Optional

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.interpreter import NaturalLanguageInterpreter
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.training.structures import StoryGraph


def extract_story_graph(
    resource_name: Text,
    domain: 'Domain',
    interpreter: Optional['NaturalLanguageInterpreter'] = None,
    use_e2e: bool = False,
    exclusion_percentage: int = None
) -> 'StoryGraph':
    from rasa_core.interpreter import RegexInterpreter
    from rasa_core.training.dsl import StoryFileReader
    from rasa_core.training.structures import StoryGraph

    if not interpreter:
        interpreter = RegexInterpreter()
    story_steps = StoryFileReader.read_from_folder(
        resource_name,
        domain, interpreter,
        use_e2e=use_e2e,
        exclusion_percentage=exclusion_percentage)
    return StoryGraph(story_steps)


def load_data(
    resource_name: Text,
    domain: 'Domain',
    remove_duplicates: bool = True,
    unique_last_num_states: Optional[int] = None,
    augmentation_factor: int = 20,
    tracker_limit: Optional[int] = None,
    use_story_concatenation: bool = True,
    debug_plots=False,
    exclusion_percentage: int = None
) -> List['DialogueStateTracker']:
    from rasa_core.training import extract_story_graph
    from rasa_core.training.generator import TrainingDataGenerator

    if resource_name:
        graph = extract_story_graph(resource_name, domain,
                                    exclusion_percentage=exclusion_percentage)

        g = TrainingDataGenerator(graph, domain,
                                  remove_duplicates,
                                  unique_last_num_states,
                                  augmentation_factor,
                                  tracker_limit,
                                  use_story_concatenation,
                                  debug_plots)
        return g.generate()
    else:
        return []


def persist_data(trackers: List['DialogueStateTracker'], path: Text) -> None:
    """Dump a list of dialogue trackers in the story format to disk."""

    for t in trackers:
        t.export_stories_to_file(path)
