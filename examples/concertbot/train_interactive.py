import asyncio
import logging

import rasa.utils
from rasa.core import utils
from rasa.core.training import interactive

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    rasa.utils.configure_colored_logging(loglevel="INFO")
    loop = asyncio.get_event_loop()
    logger.info("This example does not include NLU data."
                "Please specify the desired intent with a preceding '/', e.g."
                "'/greet' .")
    loop.run_until_complete(interactive.run_interactive_learning(
        "data/stories.md",
        server_args={
            "domain": "domain.yml",
            "out": "models/dialogue",
            "stories": "data/stories.md",
            "config": ['config.yml']
        }))
