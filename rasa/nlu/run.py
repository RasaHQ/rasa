import asyncio
import json
import logging

from rasa.cli.utils import print_success

from rasa.nlu.model import Interpreter
from rasa.core.interpreter import RegexInterpreter, INTENT_MESSAGE_PREFIX

logger = logging.getLogger(__name__)


def run_cmdline(model_path, component_builder=None):
    interpreter = Interpreter.load(model_path, component_builder)
    regex_interpreter = RegexInterpreter()

    print_success("NLU model loaded. Type a message and press enter to parse it.")
    while True:
        print_success("Next message:")
        message = input()
        if message.startswith(INTENT_MESSAGE_PREFIX):
            text = message.rstrip()
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(regex_interpreter.parse(text))
        else:
            text = message.strip()
            result = interpreter.parse(text)

        print (json.dumps(result, indent=2))


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.nlu.run` directly is no longer supported. "
        "Please use `rasa run` to start a Rasa server or `rasa shell` to use your "
        "NLU model to interpret text via the command line."
    )
