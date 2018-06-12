from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
from threading import Thread

from builtins import str
from gevent.pywsgi import WSGIServer

from rasa_core import utils, server
from rasa_core.actions.action import ActionEndpointConfig
from rasa_core.channels import RestInput, console
from rasa_core.channels.facebook import FacebookInput
from rasa_core.channels.mattermost import MattermostInput
from rasa_core.channels.slack import SlackInput
from rasa_core.channels.telegram import TelegramInput
from rasa_core.channels.twilio import TwilioInput
from rasa_core.constants import DEFAULT_SERVER_PORT, DEFAULT_SERVER_URL
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
            default=DEFAULT_SERVER_PORT,
            type=int,
            help="port to run the server at")
    parser.add_argument(
            '--auth_token',
            type=str,
            help="Enable token based authentication. Requests need to provide "
                 "the token to be accepted.")
    parser.add_argument(
            '--cors',
            nargs='*',
            type=str,
            help="enable CORS for the passed origin. "
                 "Use * to whitelist all origins")
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
            '--action_endpoint_url',
            default=None,
            help="url of the action endpoint")
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


def _create_external_channel(channel, credentials_file):
    # the commandline input channel is the only one that doesn't need any
    # credentials
    if channel == "cmdline":
        return RestInput()

    if credentials_file is None:
        _raise_missing_credentials_exception(channel)

    credentials = read_yaml_file(credentials_file)

    if channel == "facebook":
        return FacebookInput(
                credentials.get("verify"),
                credentials.get("secret"),
                credentials.get("page-access-token"))
    elif channel == "slack":
        return SlackInput(
                credentials.get("slack_token"),
                credentials.get("slack_channel"))
    elif channel == "telegram":
        return TelegramInput(
                credentials.get("access_token"),
                credentials.get("verify"),
                credentials.get("webhook_url"))
    elif channel == "mattermost":
        return MattermostInput(
                credentials.get("url"),
                credentials.get("team"),
                credentials.get("user"),
                credentials.get("pw"))
    elif channel == "twilio":
        return TwilioInput(
                credentials.get("account_sid"),
                credentials.get("auth_token"),
                credentials.get("twilio_number"))
    else:
        Exception("This script currently only supports the facebook,"
                  " telegram, mattermost and slack connectors.")


def create_http_input_channel(channel, credentials_file):
    """Instantiate the chosen input channel."""

    if channel in ['facebook', 'slack', 'telegram', 'mattermost', 'twilio', 'cmdline']:
        return _create_external_channel(channel, credentials_file)
    else:
        try:
            c = utils.class_from_module_path(channel)
            return c()
        except Exception:
            raise Exception("Unknown input channel for running main.")


def start_cmdline_io(server_url, on_finish):

    p = Thread(target=console.record_messages,
               kwargs={
                   "server_url": server_url,
                   "on_finish": on_finish})
    p.start()


def start_server(model_directory, nlu_model=None, channel=None, port=None,
                 credentials_file=None, cors=None):
    server_config = {
        "action_callback": ActionEndpointConfig(
                url="http://localhost:5055/webhook"),
        "nlg": {"type": "template"}
    }

    action_endpoint = server_config.get("action_callback")

    input_channel = create_http_input_channel(channel, credentials_file)
    app = server.create_app(model_directory,
                            nlu_model,
                            [input_channel],
                            cors,
                            auth_token=cmdline_args.auth_token,
                            action_endpoint=action_endpoint,
                            nlg_config=server_config.get("nlg"))

    http_server = WSGIServer(('0.0.0.0', cmdline_args.port), app)
    logger.info("Rasa Core server is up and running on "
                "{}".format(DEFAULT_SERVER_URL))
    http_server.start()

    if channel == "cmdline":
        start_cmdline_io(DEFAULT_SERVER_URL, http_server.stop)

    try:
        http_server.serve_forever()
    except Exception as exc:
        logger.exception(exc)


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARN)

    utils.configure_colored_logging(cmdline_args.loglevel)
    utils.configure_file_logging(cmdline_args.loglevel,
                                 cmdline_args.log_file)

    logger.info("Rasa process starting")

    start_server(cmdline_args.core,
                 cmdline_args.nlu,
                 cmdline_args.connector,
                 cmdline_args.port,
                 cmdline_args.credentials,
                 cmdline_args.cors)
