from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
from collections import namedtuple

from flask import Flask
from flask_cors import CORS
from threading import Thread

from builtins import str
from gevent.pywsgi import WSGIServer
from typing import Text, Optional, Union, List

import rasa_core
from rasa_core import constants, agent
from rasa_core import utils, server
from rasa_core.agent import Agent
from rasa_core.channels import (
    console, RestInput, InputChannel,
    BUILTIN_CHANNELS)
from rasa_core.constants import DOCS_BASE_URL
from rasa_core.interpreter import (
    NaturalLanguageInterpreter)
from rasa_core.utils import read_yaml_file

logger = logging.getLogger()  # get the root logger

AvailableEndpoints = namedtuple('AvailableEndpoints', 'nlg '
                                                      'nlu '
                                                      'action '
                                                      'model')


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
            '--endpoints',
            default=None,
            help="Configuration file for the connectors as a yml file")
    parser.add_argument(
            '-c', '--connector',
            default="cmdline",
            choices=["facebook", "slack", "telegram", "mattermost", "cmdline",
                     "twilio", "botframework", "rocketchat"],
            help="service to connect to")
    parser.add_argument(
            '--enable_api',
            action="store_true",
            help="Start the web server api in addition to the input channel")

    utils.add_logging_option_arguments(parser)
    return parser


def read_endpoints(endpoint_file):
    nlg = utils.read_endpoint_config(endpoint_file,
                                     endpoint_type="nlg")
    nlu = utils.read_endpoint_config(endpoint_file,
                                     endpoint_type="nlu")
    action = utils.read_endpoint_config(endpoint_file,
                                        endpoint_type="action_endpoint")
    model = utils.read_endpoint_config(endpoint_file,
                                       endpoint_type="models")

    return AvailableEndpoints(nlg, nlu, action, model)


def _raise_missing_credentials_exception(channel):
    raise Exception("To use the {} input channel, you need to "
                    "pass a credentials file using '--credentials'. "
                    "The argument should be a file path pointing to"
                    "a yml file containing the {} authentication"
                    "information. Details in the docs: "
                    "{}/connectors/#{}-setup".
                    format(channel, channel, DOCS_BASE_URL, channel))


def _create_external_channels(channel, credentials_file):
    # type: (Optional[Text], Optional[Text]) -> List[InputChannel]

    # the commandline input channel is the only one that doesn't need any
    # credentials
    if channel == "cmdline":
        from rasa_core.channels import RestInput
        return [RestInput()]

    if channel is None and credentials_file is None:
        # if there is no configuration at all, we'll run without a channel
        return []
    elif credentials_file is None:
        # if there is a channel, but no configuration, this can't be right
        _raise_missing_credentials_exception(channel)

    all_credentials = read_yaml_file(credentials_file)

    if channel:
        return [_create_single_channel(channel, all_credentials.get(channel))]
    else:
        return [_create_single_channel(c, k)
                for c, k in all_credentials.items()]


def _create_single_channel(channel, credentials):
    if channel == "facebook":
        from rasa_core.channels.facebook import FacebookInput

        return FacebookInput(
                credentials.get("verify"),
                credentials.get("secret"),
                credentials.get("page-access-token"))
    elif channel == "slack":
        from rasa_core.channels.slack import SlackInput

        return SlackInput(
                credentials.get("slack_token"),
                credentials.get("slack_channel"))
    elif channel == "telegram":
        from rasa_core.channels.telegram import TelegramInput

        return TelegramInput(
                credentials.get("access_token"),
                credentials.get("verify"),
                credentials.get("webhook_url"))
    elif channel == "mattermost":
        from rasa_core.channels.mattermost import MattermostInput

        return MattermostInput(
                credentials.get("url"),
                credentials.get("team"),
                credentials.get("user"),
                credentials.get("pw"))
    elif channel == "twilio":
        from rasa_core.channels.twilio import TwilioInput

        return TwilioInput(
                credentials.get("account_sid"),
                credentials.get("auth_token"),
                credentials.get("twilio_number"))
    elif channel == "botframework":
        from rasa_core.channels.botframework import BotFrameworkInput
        return BotFrameworkInput(
                credentials.get("app_id"),
                credentials.get("app_password"))
    elif channel == "rocketchat":
        from rasa_core.channels.rocketchat import RocketChatInput
        return RocketChatInput(
                credentials.get("user"),
                credentials.get("password"),
                credentials.get("server_url"))
    elif channel == "rasa":
        from rasa_core.channels.rasa_chat import RasaChatInput

        return RasaChatInput(
                credentials.get("url"),
                credentials.get("admin_token"))
    else:
        raise Exception("This script currently only supports the "
                        "{} connectors."
                        "".format(", ".join(BUILTIN_CHANNELS)))


def create_http_input_channels(channel,  # type: Union[None, Text, RestInput]
                               credentials_file  # type: Optional[Text]
                               ):
    # type: (...) -> List[InputChannel]
    """Instantiate the chosen input channel."""

    if channel is None or channel in rasa_core.channels.BUILTIN_CHANNELS:
        return _create_external_channels(channel, credentials_file)
    else:
        try:
            c = utils.class_from_module_path(channel)
            return [c()]
        except Exception:
            raise Exception("Unknown input channel for running main.")


def start_cmdline_io(server_url, on_finish, **kwargs):
    kwargs["server_url"] = server_url
    kwargs["on_finish"] = on_finish

    p = Thread(target=console.record_messages,
               kwargs=kwargs)
    p.start()


def start_server(input_channels,
                 cors,
                 auth_token,
                 port,
                 initial_agent,
                 enable_api=True):
    """Run the agent."""

    if enable_api:
        app = server.create_app(initial_agent,
                                cors_origins=cors,
                                auth_token=auth_token)
    else:
        app = Flask(__name__)
        CORS(app, resources={r"/*": {"origins": cors or ""}})

    if input_channels:
        rasa_core.channels.channel.register(input_channels,
                                            app,
                                            initial_agent.handle_message,
                                            route="/webhooks/")

    if logger.isEnabledFor(logging.DEBUG):
        utils.list_routes(app)

    http_server = WSGIServer(('0.0.0.0', port), app)
    logger.info("Rasa Core server is up and running on "
                "{}".format(constants.DEFAULT_SERVER_URL))
    http_server.start()
    return http_server


def serve_application(initial_agent,
                      channel=None,
                      port=constants.DEFAULT_SERVER_PORT,
                      credentials_file=None,
                      cors=None,
                      auth_token=None,
                      enable_api=True
                      ):
    input_channels = create_http_input_channels(channel, credentials_file)

    http_server = start_server(input_channels, cors, auth_token,
                               port, initial_agent, enable_api)

    if channel == "cmdline":
        start_cmdline_io(constants.DEFAULT_SERVER_URL, http_server.stop)

    try:
        http_server.serve_forever()
    except Exception as exc:
        logger.exception(exc)


def load_agent(core_model, interpreter, endpoints,
               tracker_store=None,
               wait_time_between_pulls=100):
    if endpoints.model:
        return agent.load_from_server(
                interpreter=interpreter,
                generator=endpoints.nlg,
                action_endpoint=endpoints.action,
                model_server=endpoints.model,
                tracker_store=tracker_store,
                wait_time_between_pulls=wait_time_between_pulls
        )
    else:
        return Agent.load(core_model,
                          interpreter=interpreter,
                          generator=endpoints.nlg,
                          tracker_store=tracker_store,
                          action_endpoint=endpoints.action)


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    logging.getLogger('werkzeug').setLevel(logging.WARN)
    logging.getLogger('matplotlib').setLevel(logging.WARN)

    utils.configure_colored_logging(cmdline_args.loglevel)
    utils.configure_file_logging(cmdline_args.loglevel,
                                 cmdline_args.log_file)

    logger.info("Rasa process starting")

    _endpoints = read_endpoints(cmdline_args.endpoints)
    _interpreter = NaturalLanguageInterpreter.create(cmdline_args.nlu,
                                                     _endpoints.nlu)
    _agent = load_agent(cmdline_args.core,
                        interpreter=_interpreter,
                        endpoints=_endpoints)

    serve_application(_agent,
                      cmdline_args.connector,
                      cmdline_args.port,
                      cmdline_args.credentials,
                      cmdline_args.cors,
                      cmdline_args.auth_token,
                      cmdline_args.enable_api)
