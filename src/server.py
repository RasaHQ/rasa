import argparse
import logging
import os
from functools import wraps

from flask import Flask
from flask import current_app
from flask import json
from flask import jsonify
from flask import request
from gevent.wsgi import WSGIServer

from rasa_nlu.config import RasaNLUConfig, DEFAULT_CONFIG_LOCATION
from rasa_nlu.data_router import DataRouter
from rasa_nlu.utils import mitie
from rasa_nlu.utils import spacy

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


def __create_interpreter(config):
    def load_model_from_s3(model_dir):
        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(config)
            p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
        except Exception as e:
            logging.warn("Using default interpreter, couldn't fetch model: {}".format(e.message))

    model_dir = config['server_model_dir']
    metadata, backend = None, None

    if model_dir is not None:
        # download model from S3 if needed
        if not os.path.isdir(model_dir):
            load_model_from_s3(model_dir)

        with open(os.path.join(model_dir, 'metadata.json'), 'rb') as meta_file:
            metadata = json.loads(meta_file.read())
        backend = metadata.get("backend")
    elif config['backend']:
        logging.warn("backend '%s' specified in config, but no model directory ('server_model_dir') is configured. " +
                     "Using 'hello-goodby' backend instead!", config['backend'])

    if backend is None:
        from interpreters.simple_interpreter import HelloGoodbyeInterpreter
        logging.info("using default hello-goodby backend")
        return HelloGoodbyeInterpreter()
    elif backend.lower() == mitie.MITIE_BACKEND_NAME:
        logging.info("using mitie backend")
        from interpreters.mitie_interpreter import MITIEInterpreter
        return MITIEInterpreter(**metadata)
    elif backend.lower() == mitie.MITIE_SKLEARN_BACKEND_NAME:
        logging.info("using mitie_sklearn backend")
        from interpreters.mitie_sklearn_interpreter import MITIESklearnInterpreter
        return MITIESklearnInterpreter(**metadata)
    elif backend.lower() == spacy.SPACY_BACKEND_NAME:
        logging.info("using spacy + sklearn backend")
        from interpreters.spacy_sklearn_interpreter import SpacySklearnInterpreter
        return SpacySklearnInterpreter(**metadata)
    else:
        raise ValueError("unknown backend : {0}".format(backend))


def __create_emulator(config):
    mode = config['emulate']
    if mode is None:
        from emulators import NoEmulator
        return NoEmulator()
    elif mode.lower() == 'wit':
        from emulators.wit import WitEmulator
        return WitEmulator()
    elif mode.lower() == 'luis':
        from emulators.luis import LUISEmulator
        return LUISEmulator()
    elif mode.lower() == 'api':
        from emulators.api import ApiEmulator
        return ApiEmulator()
    else:
        raise ValueError("unknown mode : {0}".format(mode))


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
        response = current_app.data_router.parse(data["text"])
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
    emulator = __create_emulator(config)
    interpreter = __create_interpreter(config)
    app.data_router = DataRouter(config, interpreter, emulator)
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
