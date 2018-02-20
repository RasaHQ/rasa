from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os

import six
from builtins import input

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Interpreter

logger = logging.getLogger(__name__)


def create_argparser():  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(
            description='run a Rasa NLU model locally on the command line '
                        'for manual testing')

    parser.add_argument('-c', '--config', required=True,
                        help="config file")

    parser.add_argument('-m', '--model', required=True,
                        help="path to model")
    return parser


def run_cmdline(config, model_path, component_builder=None):
    interpreter = Interpreter.load(model_path, config, component_builder)

    logger.info("NLU model loaded. Type a message and press enter to parse it.")
    while True:
        text = input().strip()
        if six.PY2:
            # in python 2 input doesn't return unicode values
            text = text.decode("utf-8")
        r = interpreter.parse(text)
        print(json.dumps(r, indent=2))


if __name__ == '__main__':  # pragma: no cover
    parser = create_argparser()
    args = parser.parse_args()

    nlu_config = RasaNLUConfig(args.config, os.environ, vars(args))
    logging.basicConfig(level=nlu_config['log_level'])

    run_cmdline(nlu_config, args.model)
