import json
import logging

from rasa.nlu import utils
from rasa.nlu.model import Interpreter

logger = logging.getLogger(__name__)


def create_argument_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="run a Rasa NLU model locally on the command line "
        "for manual testing"
    )

    parser.add_argument("-m", "--model", required=True, help="path to model")

    utils.add_logging_option_arguments(parser, default=logging.INFO)

    return parser


def run_cmdline(model_path, component_builder=None):
    interpreter = Interpreter.load(model_path, component_builder)

    logger.info("NLU model loaded. Type a message and press enter to parse it.")
    while True:
        text = input().strip()
        r = interpreter.parse(text)
        print (json.dumps(r, indent=2))
        logger.info("Next message:")


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.nlu.server` directly is "
        "no longer supported. "
        "Please use `rasa shell` instead."
    )
