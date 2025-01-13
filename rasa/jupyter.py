import asyncio
import pprint as pretty_print
import typing
from typing import Any, Dict, Optional, Text

import rasa.core.agent
import rasa.utils.common
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.cli import print_success

if typing.TYPE_CHECKING:
    from rasa.core.agent import Agent


def pprint(obj: Any) -> None:
    """Prints JSONs with indent."""
    pretty_print.pprint(obj, indent=2)


def chat(
    model_path: Optional[Text] = None,
    endpoints: Optional[Text] = None,
    agent: Optional["Agent"] = None,
) -> None:
    """Chat to the bot within a Jupyter notebook.

    Args:
        model_path: Path to a combined Rasa model.
        endpoints: Path to a yaml with the action server is custom actions are defined.
        agent: Rasa Core agent (used if no Rasa model given).
    """
    if model_path:
        agent = asyncio.run(
            rasa.core.agent.load_agent(model_path=model_path, endpoints=endpoints)
        )

    if agent is None:
        raise RasaException(
            "Either the provided model path could not load the agent "
            "or no core agent was provided."
        )

    print("Your bot is ready to talk! Type your messages here or send '/stop'.")
    while True:
        message = input()
        if message == "/stop":
            break

        responses = asyncio.run(agent.handle_text(message))
        for response in responses:
            _display_bot_response(response)


def _display_bot_response(response: Dict) -> None:
    from IPython.display import Image, display

    for response_type, value in response.items():
        if response_type == "text":
            print_success(value)

        if response_type == "image":
            image = Image(url=value)
            display(image)
