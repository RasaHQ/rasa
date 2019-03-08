import asyncio
import logging

from rasa_core import utils, train
from rasa_core.training import interactive

logger = logging.getLogger(__name__)


async def train_agent():
    return await train(domain_file="domain.yml",
                       stories_file="data/stories.md",
                       output_path="models/dialogue",
                       policy_config='policy_config.yml')


if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")
    loop = asyncio.get_event_loop()
    agent = loop.run_until_complete(train_agent())
    logger.info("This example does not include NLU data."
                "Please specify the desired intent with a preceding '/', e.g."
                "'/greet' .")
    loop.run_until_complete(interactive.run_interactive_learning(agent))
