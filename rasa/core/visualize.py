import logging
import os
from typing import Text

from rasa import telemetry
from rasa.shared.core.domain import Domain, InvalidDomain
from rasa.shared.core.training_data import loading
from rasa.shared.utils.cli import print_error

logger = logging.getLogger(__name__)


def visualize(
    domain_path: Text,
    stories_path: Text,
    nlu_data_path: Text,
    output_path: Text,
    max_history: int,
) -> None:
    """Visualizes stories as graph.

    Args:
        domain_path: Path to the domain file.
        stories_path: Path to the stories files.
        nlu_data_path: Path to the NLU training data which can be used to interpolate
            intents with actual examples in the graph.
        output_path: Path where the created graph should be persisted.
        max_history: Max history to use for the story visualization.
    """
    import rasa.shared.core.training_data.visualization

    try:
        domain = Domain.load(domain_path)
    except InvalidDomain as e:
        print_error(
            f"Could not load domain due to: '{e}'. To specify a valid domain path use "
            f"the '--domain' argument."
        )
        return

    # this is optional, only needed if the `/greet` type of
    # messages in the stories should be replaced with actual
    # messages (e.g. `hello`)
    if nlu_data_path is not None:
        import rasa.shared.nlu.training_data.loading

        nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
            nlu_data_path
        )
    else:
        nlu_training_data = None

    logger.info("Starting to visualize stories...")
    telemetry.track_visualization()

    story_steps = loading.load_data_from_resource(stories_path, domain)
    rasa.shared.core.training_data.visualization.visualize_stories(
        story_steps,
        domain,
        output_path,
        max_history,
        nlu_training_data=nlu_training_data,
    )

    full_output_path = "file://{}".format(os.path.abspath(output_path))
    logger.info(f"Finished graph creation. Saved into {full_output_path}")

    import webbrowser

    webbrowser.open(full_output_path)
