from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

from builtins import str
from typing import Optional, Union, Text

from rasa_core import utils, constants
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.channels.facebook import FacebookInput
from rasa_core.channels.telegram import TelegramInput
from rasa_core.channels.rest import HttpInputChannel
from rasa_core.channels.slack import SlackInput
from rasa_core.channels.mattermost import MattermostInput
from rasa_core.channels.twilio import TwilioInput
from rasa_core.interpreter import (
    NaturalLanguageInterpreter,
    RasaNLUHttpInterpreter)
from rasa_core.utils import read_yaml_file, EndpointConfig

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
            default=constants.DEFAULT_SERVER_PORT,
            type=int,
            help="port to run the server at (if a server is run "
                 "- depends on the chosen channel, e.g. facebook uses this)")
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
            '--endpoints',
            default=None,
            help="Configuration file for the connectors as a yml file")
    parser.add_argument(
            '-c', '--connector',
            default="cmdline",
            choices=["facebook", "slack", "telegram", "mattermost", "cmdline",
                     "twilio"],
            help="service to connect to")

    utils.add_logging_option_arguments(parser)
    return parser


def _raise_missing_credentials_exception(channel):
    if channel == "facebook":
        channel_doc_link = "facebook-messenger"
    elif channel == "slack":
        channel_doc_link = "slack"
    elif channel == "telegram":
        channel_doc_link = "telegram"
    elif channel == "mattermost":
        channel_doc_link = "mattermost"
    elif channel == "twilio":
        channel_doc_link = "twilio"
    else:
        channel_doc_link = ""

    raise Exception("To use the {} input channel, you need to "
                    "pass a credentials file using '--credentials'. "
                    "The argument should be a file path pointing to"
                    "a yml file containing the {} authentication"
                    "information. Details in the docs: "
                    "https://core.rasa.com/connectors.html#{}-setup".
                    format(channel, channel, channel_doc_link))


def _create_external_channel(channel, port, credentials_file):
    if credentials_file is None:
        _raise_missing_credentials_exception(channel)

    credentials = read_yaml_file(credentials_file)
    if channel == "facebook":
        input_blueprint = FacebookInput(
                credentials.get("verify"),
                credentials.get("secret"),
                credentials.get("page-access-token"))
    elif channel == "slack":
        input_blueprint = SlackInput(
                credentials.get("slack_token"),
                credentials.get("slack_channel"))
    elif channel == "telegram":
        input_blueprint = TelegramInput(
                credentials.get("access_token"),
                credentials.get("verify"),
                credentials.get("webhook_url"))
    elif channel == "mattermost":
        input_blueprint = MattermostInput(
                credentials.get("url"),
                credentials.get("team"),
                credentials.get("user"),
                credentials.get("pw"))
    elif channel == "twilio":
        input_blueprint = TwilioInput(
                credentials.get("account_sid"),
                credentials.get("auth_token"),
                credentials.get("twilio_number"))
    else:
        Exception("This script currently only supports the facebook,"
                  " telegram, mattermost and slack connectors.")

    return HttpInputChannel(port, None, input_blueprint)


def create_input_channel(channel, port, credentials_file):
    """Instantiate the chosen input channel."""

    if channel in ['facebook', 'slack', 'telegram', 'mattermost', 'twilio']:
        return _create_external_channel(channel, port, credentials_file)
    elif channel == "cmdline":
        return ConsoleInputChannel()
    else:
        try:
            c = utils.class_from_module_path(channel)
            return c()
        except Exception:
            raise Exception("Unknown input channel for running main.")


def interpreter_from_args(
        nlu_model,  # type: Union[Text, NaturalLanguageInterpreter, None]
        nlu_endpoint  # type: Optional[EndpointConfig]
        ):
    # type: (...) -> Optional[NaturalLanguageInterpreter]
    """Create an interpreter from the commandline arguments.

    Depending on which values are passed for model and endpoint, this
    will create the corresponding interpreter (either loading the model
    locally or setting up an endpoint based interpreter)."""

    if isinstance(nlu_model, NaturalLanguageInterpreter):
        return nlu_model

    if nlu_model:
        name_parts = nlu_model.split("/")
    else:
        name_parts = []

    if len(name_parts) == 1:
        if nlu_endpoint:
            # using the default project name
            return RasaNLUHttpInterpreter(name_parts[0],
                                          nlu_endpoint)
        else:
            return NaturalLanguageInterpreter.create(nlu_model)
    elif len(name_parts) == 2:
        if nlu_endpoint:
            return RasaNLUHttpInterpreter(name_parts[0],
                                          nlu_endpoint,
                                          name_parts[1])
        else:
            return NaturalLanguageInterpreter.create(nlu_model)
    else:
        if nlu_endpoint:
            raise Exception("You have configured an endpoint to use for "
                            "the NLU model. To use it, you need to "
                            "specify the model to use with "
                            "`--nlu project/model`.")
        else:
            return None


def main(model_directory, nlu_model=None, channel=None, port=None,
         credentials_file=None, nlg_endpoint=None, nlu_endpoint=None):
    """Run the agent."""

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARN)

    logger.info("Rasa process starting")

    interpreter = interpreter_from_args(nlu_model, nlu_endpoint)
    agent = Agent.load(model_directory, interpreter,
                       generator=nlg_endpoint)

    logger.info("Finished loading agent, starting input channel & server.")
    if channel:
        input_channel = create_input_channel(channel, port, credentials_file)
        agent.handle_channel(input_channel)

    return agent


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)
    utils.configure_file_logging(cmdline_args.loglevel,
                                 cmdline_args.log_file)

    nlg_endpoint = utils.read_endpoint_config(cmdline_args.endpoints,
                                              "nlg")

    nlu_endpoint = utils.read_endpoint_config(cmdline_args.endpoints,
                                              "nlu")

    main(cmdline_args.core,
         cmdline_args.nlu,
         cmdline_args.connector,
         cmdline_args.port,
         cmdline_args.credentials,
         nlg_endpoint,
         nlu_endpoint)
