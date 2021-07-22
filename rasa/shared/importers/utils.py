import asyncio
from typing import Coroutine, Any, Iterable, Text, Optional, Dict, List, TypeVar

from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.training_data.training_data import TrainingData

T = TypeVar("T")


# TODO(alwx): temporary solution
def run_in_loop(
    f: Coroutine[Any, Any, T], loop: Optional[asyncio.AbstractEventLoop] = None
) -> T:
    """Execute the awaitable in the passed loop.

    If no loop is passed, the currently existing one is used or a new one is created
    if no loop has been started in the current context.

    After the awaitable is finished, all remaining tasks on the loop will be
    awaited as well (background tasks).

    WARNING: don't use this if there are never ending background tasks scheduled.
        in this case, this function will never return.

    Args:
       f: function to execute
       loop: loop to use for the execution

    Returns:
        return value from the function
    """

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    result = loop.run_until_complete(f)

    # Let's also finish all running tasks:
    # fallback for python 3.6, which doesn't have asyncio.all_tasks
    try:
        pending = asyncio.all_tasks(loop)
    except AttributeError:
        pending = asyncio.Task.all_tasks()
    loop.run_until_complete(asyncio.gather(*pending))

    return result


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    from rasa.shared.nlu.training_data import loading

    training_data_sets = [loading.load_data(nlu_file, language) for nlu_file in paths]
    return TrainingData().merge(*training_data_sets)


async def story_graph_from_paths(
    files: List[Text],
    domain: Domain,
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> StoryGraph:

    from rasa.shared.core.training_data import loading

    story_steps = await loading.load_data_from_files(
        files, domain, template_variables, use_e2e, exclusion_percentage
    )
    return StoryGraph(story_steps)
