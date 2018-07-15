from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import tempfile
import zipfile
from functools import wraps

from builtins import str
from flask import Flask, request, abort, Response, jsonify
from flask_cors import CORS, cross_origin
from gevent.pywsgi import WSGIServer
from typing import Union, Text, Optional

from rasa_core import utils, events, run
from rasa_core.agent import Agent
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.tracker_store import TrackerStore
from rasa_core.trackers import DialogueStateTracker
from rasa_core.version import __version__

from typing import Union
import typing

if typing.TYPE_CHECKING:
    from rasa_core.interpreter import NaturalLanguageInterpreter as NLI

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the server script."""

    parser = argparse.ArgumentParser(
            description='starts server to serve an agent')
    parser.add_argument(
            '-d', '--core',
            required=True,
            type=str,
            help="core model to run with the server")
    parser.add_argument(
            '-u', '--nlu',
            type=str,
            help="nlu model to run with the server")
    parser.add_argument(
            '-p', '--port',
            type=int,
            default=5005,
            help="port to run the server at")
    parser.add_argument(
            '--cors',
            nargs='*',
            type=str,
            help="enable CORS for the passed origin. "
                 "Use * to whitelist all origins")
    parser.add_argument(
            '--auth_token',
            type=str,
            help="Enable token based authentication. Requests need to provide "
                 "the token to be accepted.")
    parser.add_argument(
            '-o', '--log_file',
            type=str,
            default="rasa_core.log",
            help="store log file in specified file")
    parser.add_argument(
            '--endpoints',
            default=None,
            help="Configuration file for the connectors as a yml file")

    utils.add_logging_option_arguments(parser)
    return parser


def ensure_loaded_agent(agent):
    """Wraps a request handler ensuring there is a loaded and usable model."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            __agent = agent()
            if not __agent:
                return Response(
                        "No agent loaded. To continue processing, a model "
                        "of a trained agent needs to be loaded.",
                        status=503)

            return f(*args, **kwargs)

        return decorated

    return decorator


def bool_arg(name, default=True):
    # type: ( Text, bool) -> bool
    """Return a passed boolean argument of the request or a default.

    Checks the `name` parameter of the request if it contains a valid
    boolean value. If not, `default` is returned."""

    return request.args.get(name, str(default)).lower() == 'true'


def request_parameters():
    if request.method == 'GET':
        return request.args
    else:

        try:
            return request.get_json(force=True)
        except ValueError as e:
            logger.error("Failed to decode json during respond request. "
                         "Error: {}.".format(e))
            raise


def requires_auth(token=None):
    # type: (Optional[Text]) -> function
    """Wraps a request handler with token authentication."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            provided = request.args.get('token')
            if token is None or provided == token:
                return f(*args, **kwargs)
            abort(401)

        return decorated

    return decorator


def _create_agent(
        model_directory,  # type: Text
        interpreter,  # type: Union[Text,NLI,None]
        action_factory=None,  # type: Optional[Text]
        tracker_store=None,  # type: Optional[TrackerStore]
        generator=None
):
    # type: (...) -> Optional[Agent]
    try:

        return Agent.load(model_directory, interpreter,
                          tracker_store=tracker_store,
                          action_factory=action_factory,
                          generator=generator)
    except Exception as e:
        logger.warn("Failed to load any agent model. Running "
                    "Rasa Core server with out loaded model now. {}"
                    "".format(e))
        return None


def create_app(model_directory,  # type: Text
               interpreter=None,  # type: Union[Text, NLI, None]
               loglevel="INFO",  # type: Optional[Text]
               logfile="rasa_core.log",  # type: Optional[Text]
               cors_origins=None,  # type: Optional[List[Text]]
               action_factory=None,  # type: Optional[Text]
               auth_token=None,  # type: Optional[Text]
               tracker_store=None,  # type: Optional[TrackerStore]
               endpoints=None
               ):
    """Class representing a Rasa Core HTTP server."""

    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    # Setting up logfile
    utils.configure_file_logging(loglevel, logfile)

    if not cors_origins:
        cors_origins = []
    model_directory = model_directory

    nlg_endpoint = utils.read_endpoint_config(endpoints, "nlg")

    nlu_endpoint = utils.read_endpoint_config(endpoints, "nlu")

    tracker_store = tracker_store

    action_factory = action_factory

    _interpreter = run.interpreter_from_args(interpreter, nlu_endpoint)

    # this needs to be an array, so we can modify it in the nested functions...
    _agent = [_create_agent(model_directory, _interpreter,
                            action_factory, tracker_store, nlg_endpoint)]

    def agent():
        if _agent and _agent[0]:
            return _agent[0]
        else:
            return None

    @app.route("/",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    def hello():
        """Check if the server is running and responds with the version."""
        return "hello from Rasa Core: " + __version__

    @app.route("/version",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    def version():
        """respond with the version number of the installed rasa core."""

        return jsonify({'version': __version__})

    # <sender_id> can be be 'default' if there's only 1 client
    @app.route("/conversations/<sender_id>/continue",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def continue_predicting(sender_id):
        """Continue a prediction started with parse.

        Caller should have executed the action returned from the parse
        endpoint. The events returned from that executed action are
        passed to continue which will trigger the next action prediction.

        If continue predicts action listen, the caller should wait for the
        next user message."""

        request_params = request.get_json(force=True)
        encoded_events = request_params.get("events", [])
        executed_action = request_params.get("executed_action", None)
        evts = events.deserialise_events(encoded_events)
        try:
            response = agent().continue_message_handling(sender_id,
                                                         executed_action,
                                                         evts)
        except ValueError as e:
            return Response(jsonify(error=e.message),
                            status=400,
                            content_type="application/json")
        except Exception as e:
            logger.exception(e)
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")
        return jsonify(response)

    @app.route("/conversations/<sender_id>/tracker/events",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def append_events(sender_id):
        """Append a list of events to the state of a conversation"""

        request_params = request.get_json(force=True)
        evts = events.deserialise_events(request_params)
        tracker = agent().tracker_store.get_or_create_tracker(sender_id)
        for e in evts:
            tracker.update(e)
        agent().tracker_store.save(tracker)
        return jsonify(tracker.current_state())

    @app.route("/conversations",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def list_trackers():
        return jsonify(list(agent().tracker_store.keys()))

    @app.route("/conversations/<sender_id>/tracker",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def retrieve_tracker(sender_id):
        """Get a dump of a conversations tracker including its events."""

        # parameters
        use_history = bool_arg('ignore_restarts', default=False)
        should_include_events = bool_arg('events', default=True)
        until_time = request.args.get('until', None)

        # retrieve tracker and set to requested state
        tracker = agent().tracker_store.get_or_create_tracker(sender_id)
        if until_time is not None:
            tracker = tracker.travel_back_in_time(float(until_time))

        # dump and return tracker
        state = tracker.current_state(
                should_include_events=should_include_events,
                only_events_after_latest_restart=use_history)
        return jsonify(state)

    @app.route("/conversations/<sender_id>/tracker",
               methods=['PUT', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def update_tracker(sender_id):
        """Use a list of events to set a conversations tracker to a state."""

        request_params = request.get_json(force=True)
        tracker = DialogueStateTracker.from_dict(sender_id,
                                                 request_params,
                                                 agent().domain)
        agent().tracker_store.save(tracker)

        # will override an existing tracker with the same id!
        agent().tracker_store.save(tracker)
        return jsonify(tracker.current_state(should_include_events=True))

    @app.route("/domain",
               methods=['GET'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def get_domain():
        """Get current domain in yaml format."""
        accepts = request.headers.get("Accept", default="application/json")
        if accepts.endswith("json"):
            domain = agent().domain.as_dict()
            return jsonify(domain)
        elif accepts.endswith("yml"):
            domain_yaml = agent().domain.as_yaml()
            return Response(domain_yaml, status=200, content_type="application/x-yml")
        else:
            return Response(
                """Invalid accept header. Domain can be provided as json ("Accept: application/json") or yml ("Accept: application/x-yml"). Make sure you've set the appropriate Accept header.""",
                status=406)


    @app.route("/conversations/<sender_id>/parse",
               methods=['GET', 'POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def parse(sender_id):
        request_params = request_parameters()

        if 'query' in request_params:
            message = request_params.pop('query')
        elif 'q' in request_params:
            message = request_params.pop('q')
        else:
            return Response(
                    jsonify(error="Invalid parse parameter specified."),
                    status=400,
                    mimetype="application/json")

        try:
            # Fetches the predicted action in a json format
            response = agent().start_message_handling(message, sender_id)
            return jsonify(response)

        except Exception as e:
            logger.exception("Caught an exception during parse.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    @app.route("/conversations/<sender_id>/respond",
               methods=['GET', 'POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def respond(sender_id):
        request_params = request_parameters()

        if 'query' in request_params:
            message = request_params.pop('query')
        elif 'q' in request_params:
            message = request_params.pop('q')
        else:
            return Response(jsonify(error="Invalid respond parameter "
                                          "specified."),
                            status=400,
                            mimetype="application/json")

        try:
            # Set the output channel
            out = CollectingOutputChannel()
            # Fetches the appropriate bot response in a json format
            responses = agent().handle_message(message,
                                               output_channel=out,
                                               sender_id=sender_id)
            return jsonify(responses)

        except Exception as e:
            logger.exception("Caught an exception during respond.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    @app.route("/load", methods=['POST', 'OPTIONS'])
    @requires_auth(auth_token)
    @cross_origin(origins=cors_origins)
    def load_model():
        """Loads a zipped model, replacing the existing one."""

        if 'model' not in request.files:
            # model file is missing
            abort(400)

        model_file = request.files['model']

        logger.info("Received new model through REST interface.")
        zipped_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        zipped_path.close()

        model_file.save(zipped_path.name)

        logger.debug("Downloaded model to {}".format(zipped_path.name))

        zip_ref = zipfile.ZipFile(zipped_path.name, 'r')
        zip_ref.extractall(model_directory)
        zip_ref.close()
        logger.debug("Unzipped model to {}".format(
                os.path.abspath(model_directory)))

        _agent[0] = _create_agent(model_directory, interpreter,
                                  action_factory, tracker_store, nlg_endpoint)
        logger.debug("Finished loading new agent.")
        return jsonify({'success': 1})

    return app


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    # Setting up the color scheme of logger
    utils.configure_colored_logging(cmdline_args.loglevel)

    # Setting up the rasa_core application framework
    app = create_app(cmdline_args.core,
                     cmdline_args.nlu,
                     cmdline_args.loglevel,
                     cmdline_args.log_file,
                     cmdline_args.cors,
                     auth_token=cmdline_args.auth_token,
                     endpoints=cmdline_args.endpoints)

    logger.info("Started http server on port %s" % cmdline_args.port)

    # Running the server at 'this' address with the
    # rasa_core application framework
    http_server = WSGIServer(('0.0.0.0', cmdline_args.port), app)
    logger.info("Up and running")
    try:
        http_server.serve_forever()
    except Exception as exc:
        logger.exception(exc)
