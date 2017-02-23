import argparse
import logging
import os
from functools import wraps

from flask import Flask
from flask import current_app
from flask import jsonify
from flask import request
from gevent.wsgi import WSGIServer

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.data_router import DataRouter


app = Flask(__name__)


def create_argparser():
    parser = argparse.ArgumentParser(description='parse incoming text')
    parser.add_argument('-c', '--config',
                        help="config file, all the command line options can also be passed via a (json-formatted) " +
                             "config file. NB command line args take precedence")
    parser.add_argument('-d', '--server_model_dir',
                        help='directory containing model to for parser to use')
    parser.add_argument('-e', '--emulate', choices=['wit', 'luis', 'api'],
                        help='which service to emulate (default: None i.e. use simple built in format)')
    parser.add_argument('-l', '--language', choices=['de', 'en'], help="model and data language")
    parser.add_argument('-m', '--mitie_file',
                        help='file with mitie total_word_feature_extractor')
    parser.add_argument('-p', '--path', help="path where model files will be saved")
    parser.add_argument('-P', '--port', type=int, help='port on which to run server')
    parser.add_argument('-t', '--token',
                        help="auth token. If set, reject requests which don't provide this token as a query parameter")
    parser.add_argument('-w', '--write', help='file where logs will be saved')

    return parser


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get('token', '')
        if current_app.data_router.token is None or token == current_app.data_router.token:
            return f(*args, **kwargs)
        return "unauthorized", 401

    return decorated


@app.route("/parse", methods=['GET', 'POST'])
@requires_auth
def parse_get():
    if request.method == 'GET':
        request_params = request.args
    else:
        request_params = request.get_json(force=True)
    if 'q' not in request_params:
        return jsonify(error="Invalid parse parameter specified")
    else:
        data = current_app.data_router.extract(request_params)
        response = current_app.data_router.parse(data)
        formatted = current_app.data_router.format_response(response)
        return jsonify(formatted)


@app.route("/status", methods=['GET'])
@requires_auth
def status():
    return jsonify(current_app.data_router.get_status())


@app.route("/", methods=['GET'])
@requires_auth
def hello():
    return "hello"


@app.route("/train", methods=['POST'])
@requires_auth
def train():
    data_string = request.get_data(as_text=True)
    current_app.data_router.start_train_proc(data_string)
    return jsonify(info="training started with pid {0}".format(current_app.data_router.train_proc.pid))


def setup_app(config):
    logging.basicConfig(filename=config['log_file'], level=config['log_level'])
    logging.captureWarnings(True)
    logging.info(config.view())

    logging.debug("Creating a new data router")
    app.data_router = DataRouter(config)
    return app


if __name__ == '__main__':
    # Running as standalone python application
    parser = create_argparser()
    cmdline_args = {key: val for key, val in vars(parser.parse_args()).items() if val is not None}
    rasa_config = RasaNLUConfig(cmdline_args.get("config"), os.environ, cmdline_args)
    http_server = WSGIServer(('0.0.0.0', rasa_config['port']), setup_app(rasa_config))
    logging.info('Started http server on port %s' % rasa_config['port'])
    http_server.serve_forever()

if __name__ == 'rasa_nlu.server':
    # Running in WSGI container, configuration will be loaded from the default location
    # There is no common support for WSGI runners to pass arguments to the application, hence we need to fallback to
    # a default location for the configuration where all the settings should be placed in.
    rasa_config = RasaNLUConfig(env_vars=os.environ)
    _app = setup_app(rasa_config)
    logging.info("Finished setting up application")
    application = _app
