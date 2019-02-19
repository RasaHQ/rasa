import json
import logging

from rasa_nlu import utils
from rasa_nlu.model import Interpreter

logger = logging.getLogger(__name__)


def create_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description='run a Rasa NLU model locally on the command line '
                    'for manual testing')

    parser.add_argument('-m', '--model', required=True,
                        help="path to model")

    utils.add_logging_option_arguments(parser, default=logging.INFO)

    return parser


def run_cmdline(model_path, component_builder=None):
    interpreter = Interpreter.load(model_path, component_builder)

    logger.info("NLU model loaded. Type a message and "
                "press enter to parse it.")
    while True:
        text = input().strip()
        r = interpreter.parse(text)
        print(json.dumps(r, indent=2))
        logger.info("Next message:")


if __name__ == '__main__':
    cmdline_args = create_argument_parser().parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    run_cmdline(cmdline_args.model)
