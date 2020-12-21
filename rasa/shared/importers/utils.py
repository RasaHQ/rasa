from typing import Iterable, Text, Optional, Dict, List

from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.training_data.training_data import TrainingDataFull


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingDataFull:
    """Loads training data from provided paths.

    Args:
        paths: paths from where to load training data.
        language: The language specified in the config.

    Returns:
        The training data.
    """
    from rasa.shared.nlu.training_data import loading

    training_data_sets = [loading.load_data(nlu_file, language) for nlu_file in paths]
    return TrainingDataFull().merge(*training_data_sets)


async def story_graph_from_paths(
    files: List[Text],
    domain: Domain,
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> StoryGraph:
    """Loads story graph from paths.

    Args:
        files: List of files with training data in it.
        domain: Domain object.
        template_variables: Variables that have to be replaced in the training data.
        use_e2e: Identifies whether the e2e reader should be used.
        exclusion_percentage: Identifies the percentage of training data that
                              should be excluded from the training.

    Returns:
        Story graph from the training data.
    """
    from rasa.shared.core.training_data import loading

    story_steps = await loading.load_data_from_files(
        files, domain, template_variables, use_e2e, exclusion_percentage
    )
    return StoryGraph(story_steps)
