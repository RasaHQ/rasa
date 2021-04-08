import logging
import os
from typing import Text

from rasa import telemetry
from rasa.shared.utils.cli import print_error

from rasa.shared.core.domain import InvalidDomain

logger = logging.getLogger(__name__)


async def visualize(
    config_path: Text,
    domain_path: Text,
    stories_path: Text,
    nlu_data_path: Text,
    output_path: Text,
    max_history: int,
) -> None:
    from rasa.core.agent import Agent
    from rasa.core import config

    try:
        policies = config.load(config_path)
    except Exception as e:
        print_error(
            f"Could not load config due to: '{e}'. To specify a valid config file use "
            f"the '--config' argument."
        )
        return

    try:
        agent = Agent(domain=domain_path, policies=policies)
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
    await agent.visualize(
        stories_path, output_path, max_history, nlu_training_data=nlu_training_data
    )

    full_output_path = "file://{}".format(os.path.abspath(output_path))
    logger.info(f"Finished graph creation. Saved into {full_output_path}")

    import webbrowser

    webbrowser.open(full_output_path)
