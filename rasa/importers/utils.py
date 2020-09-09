from typing import Iterable, Text, Optional, Dict, List

from rasa.core.domain import Domain
from rasa.core.training.structures import StoryGraph
from rasa.nlu.training_data import TrainingData


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    from rasa.nlu.training_data import loading

    training_data_sets = [loading.load_data(nlu_file, language) for nlu_file in paths]
    return TrainingData().merge(*training_data_sets)


async def story_graph_from_paths(
    files: List[Text],
    domain: Domain,
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> StoryGraph:

    from rasa.core.training import loading

    story_steps = await loading.load_data_from_files(
        files, domain, template_variables, use_e2e, exclusion_percentage
    )
    return StoryGraph(story_steps)
