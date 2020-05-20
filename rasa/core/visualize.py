import logging
import os
from typing import Text

from rasa.cli.utils import print_error

from rasa.core.domain import InvalidDomain

logger = logging.getLogger(__name__)


async def visualize(
    config_path: Text,
    domain_path: Text,
    stories_path: Text,
    nlu_data_path: Text,
    output_path: Text,
    max_history: int,
):
    from rasa.core.agent import Agent
    from rasa.core import config

    try:
        policies = config.load(config_path)
    except ValueError as e:
        print_error(
            "Could not load config due to: '{}'. To specify a valid config file use "
            "the '--config' argument.".format(e)
        )
        return

    try:
        agent = Agent(domain=domain_path, policies=policies)
    except InvalidDomain as e:
        print_error(
            "Could not load domain due to: '{}'. To specify a valid domain path use "
            "the '--domain' argument.".format(e)
        )
        return

    # this is optional, only needed if the `/greet` type of
    # messages in the stories should be replaced with actual
    # messages (e.g. `hello`)
    if nlu_data_path is not None:
        from rasa.nlu.training_data import load_data

        nlu_data_path = load_data(nlu_data_path)
    else:
        nlu_data_path = None

    logger.info("Starting to visualize stories...")
    await agent.visualize(
        stories_path, output_path, max_history, nlu_training_data=nlu_data_path
    )

    full_output_path = "file://{}".format(os.path.abspath(output_path))
    logger.info(f"Finished graph creation. Saved into {full_output_path}")

    import webbrowser

    webbrowser.open(full_output_path)


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.core.visualize` directly is "
        "no longer supported. "
        "Please use `rasa visualize` instead."
    )
