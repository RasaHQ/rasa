import asyncio
import logging
from typing import Text

from rasa.core.agent import Agent
from rasa.shared.utils.cli import print_info, print_success
from rasa.shared.utils.io import json_to_string

logger = logging.getLogger(__name__)


def run_cmdline(model_path: Text) -> None:
    """Loops over CLI input, passing each message to a loaded NLU model."""
    agent = Agent.load(model_path)

    print_success("NLU model loaded. Type a message and press enter to parse it.")
    while True:
        print_success("Next message:")
        try:
            message = input().strip()
        except (EOFError, KeyboardInterrupt):
            print_info("Wrapping up command line chat...")
            break

        result = asyncio.run(agent.parse_message(message))

        print(json_to_string(result))
