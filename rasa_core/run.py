from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

from builtins import str

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.channels.facebook import FacebookInput
from rasa_core.channels.rest import HttpInputChannel
from rasa_core.utils import read_yaml_file

logger = logging.getLogger()  # get the root logger


def create_argument_parser():
    """Parse all the command line arguments for the run script."""

    parser = argparse.ArgumentParser(
            description='starts the bot')
    parser.add_argument(
            '-d', '--core',
            required=True,
            type=str,
            help="core model to run")
    parser.add_argument(
            '-u', '--nlu',
            type=str,
            help="nlu model to run")
    parser.add_argument(
            '-p', '--port',
            default=5002,
            type=int,
            help="port to run the server at (if a server is run "
                 "- depends on the chosen channel, e.g. facebook uses this)")
    parser.add_argument(
            '-v', '--verbose',
            default=True,
            action="store_true",
            help="use verbose logging")
    parser.add_argument(
            '-o', '--log_file',
            type=str,
            default="rasa_core.log",
            help="store log file in specified file")
    parser.add_argument(
            '--credentials',
            default=None,
            help="authentication credentials for the connector as a yml file")
    parser.add_argument(
            '-c', '--connector',
            default="cmdline",
            choices=["facebook", "cmdline"],
            help="service to connect to")

    return parser


def _create_facebook_channel(channel, port, credentials_file):
    if credentials_file is None:
        raise Exception("To use the facebook input channel, you need to "
                        "pass a credentials file using '--credentials'. "
                        "The argument should be a file path pointing to"
                        "a yml file containing the facebook authentication"
                        "information. Details in the docs: "
                        "https://core.rasa.ai/facebook.html")
    credentials = read_yaml_file(credentials_file)
    input_blueprint = FacebookInput(
            credentials.get("verify"),
            credentials.get("secret"),
            credentials.get("page-tokens"),
            debug_mode=True)

    return HttpInputChannel(port, None, input_blueprint)


def create_input_channel(channel, port, credentials_file):
    """Instantiate the chosen input channel."""

    if channel == "facebook":
        return _create_facebook_channel(channel, port, credentials_file)
    elif channel == "cmdline":
        return ConsoleInputChannel()
    else:
        try:
            c = utils.class_from_module_path(channel)
            return c()
        except Exception:
            raise Exception("Unknown input channel for running main.")


def main(model_directory, nlu_model=None, channel=None, port=None,
         credentials_file=None):
    """Run the agent."""

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARN)

    logger.info("Rasa process starting")
    agent = Agent.load(model_directory, nlu_model)

    logger.info("Finished loading agent, starting input channel & server.")
    if channel:
        input_channel = create_input_channel(channel, port, credentials_file)
        agent.handle_channel(input_channel)

    return agent


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    logging.basicConfig(level="DEBUG" if cmdline_args.verbose else "INFO")

    main(cmdline_args.core,
         cmdline_args.nlu,
         cmdline_args.connector,
         cmdline_args.port,
         cmdline_args.credentials)
