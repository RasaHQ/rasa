from typing import Iterable, Text, Optional, List

from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.training_data.training_data import TrainingData


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    from rasa.shared.nlu.training_data import loading

    training_data_sets = [loading.load_data(nlu_file, language) for nlu_file in paths]
    return TrainingData().merge(*training_data_sets)


def story_graph_from_paths(
    files: List[Text], domain: Domain, exclusion_percentage: Optional[int] = None
) -> StoryGraph:
    """Returns the `StoryGraph` from paths."""
    from rasa.shared.core.training_data import loading

    story_steps = loading.load_data_from_files(files, domain, exclusion_percentage)
    return StoryGraph(story_steps)


def flows_from_paths(files: List[Text]) -> FlowsList:
    """Returns the flows from paths."""
    from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader

    flows = FlowsList(underlying_flows=[])
    for file in files:
        flows = flows.merge(YAMLFlowsReader.read_from_file(file))
    flows.validate()
    return flows
