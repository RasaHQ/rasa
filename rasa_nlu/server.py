from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import json
import logging
import os
from functools import wraps

from klein import Klein
from twisted.internet import reactor, threads

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.data_router import DataRouter, InvalidModelError
from rasa_nlu.version import __version__

logger = logging.getLogger(__name__)


def create_argparser():
    parser = argparse.ArgumentParser(description='parse incoming text')
    parser.add_argument('-c', '--config',
                        help="config file, all the command line options can also be passed via a (json-formatted) " +
                             "config file. NB command line args take precedence")
    parser.add_argument('-d', '--server_model_dirs',
                        help='directory containing model to for parser to use')
    parser.add_argument('-e', '--emulate', choices=['wit', 'luis', 'api'],
                        help='which service to emulate (default: None i.e. use simple built in format)')
    parser.add_argument('-l', '--language', choices=['de', 'en'], help="model and data language")
    parser.add_argument('-m', '--mitie_file',
                        help='file with mitie total_word_feature_extractor')
    parser.add_argument('-p', '--path', help="path where model files will be saved")
    parser.add_argument('--pipeline', help="The pipeline to use. Either a pipeline template name or a list of " +
                                           "components separated by comma")
    parser.add_argument('-P', '--port', type=int, help='port on which to run server')
    parser.add_argument('-t', '--token',
                        help="auth token. If set, reject requests which don't provide this token as a query parameter")
    parser.add_argument('-w', '--write', help='file where logs will be saved')

    return parser


def requires_auth(f):
    """Wraps a request handler with token authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        request = args[1]
        token = request.args.get('token', [''])[0]
        if self.data_router.token is None or token == self.data_router.token:
            return f(*args, **kwargs)
        return "unauthorized", 401

    return decorated


class RasaNLU(object):
    app = Klein()

    def __init__(self, config, component_builder=None):
        logging.basicConfig(filename=config['log_file'], level=config['log_level'])
        logging.captureWarnings(True)
        logger.info("Configuration: " + config.view())

        logger.debug("Creating a new data router")
        self.config = config
        self.data_router = DataRouter(config, component_builder)
        reactor.suggestThreadPoolSize(20)

    @app.route("/parse", methods=['GET', 'POST'])
    @requires_auth
    def parse_get(self, request):
        if request.method == 'GET':
            request_params = {key: value[0] for key, value in request.args.items()}
        else:
            request_params = json.loads(request.content.read())
        if 'q' not in request_params:
            request.setResponseCode(404)
            request.setHeader('Content-Type', 'application/json')
            return json.dumps({"error": "Invalid parse parameter specified"})
        else:
            try:
                data = self.data_router.extract(request_params)
                response = threads.deferToThread(self.data_router.parse, data)
                request.setHeader('Content-Type', 'application/json')
                response.addCallback(json.dumps)
                return response
            except InvalidModelError as e:
                request.setResponseCode(404)
                request.setHeader('Content-Type', 'application/json')
                return json.dumps({"error": "{}".format(e)})

    @app.route("/version", methods=['GET'])
    @requires_auth
    def version(self, request):
        request.setHeader('Content-Type', 'application/json')
        return json.dumps({'version': __version__})

    @app.route("/config", methods=['GET'])
    @requires_auth
    def rasaconfig(self, request):
        request.setHeader('Content-Type', 'application/json')
        return json.dumps(self.config.as_dict())

    @app.route("/status", methods=['GET'])
    @requires_auth
    def status(self, request):
        request.setHeader('Content-Type', 'application/json')
        return json.dumps(self.data_router.get_status())

    @app.route("/", methods=['GET'])
    def hello(self, request):
        return "hello from Rasa NLU: " + __version__

    @app.route("/train", methods=['POST'])
    @requires_auth
    def train(self, request):
        data_string = request.content.read()

        test = self.data_router.start_train_process(data_string, {key: value[0] for key, value in request.args.items()})
        test.addCallback(lambda x: print(x))

        request.setHeader('Content-Type', 'application/json')
        return json.dumps({"info": "training started.", "training_process_ids": self.data_router.train_proc_ids()})


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argparser()
    cmdline_args = {key: val for key, val in list(vars(arg_parser.parse_args()).items()) if val is not None}
    rasa_nlu_config = RasaNLUConfig(cmdline_args.get("config"), os.environ, cmdline_args)
    rasa = RasaNLU(rasa_nlu_config)
    rasa.app.run('0.0.0.0', rasa_nlu_config['port'])
    logger.info('Started http server on port %s' % rasa_nlu_config['port'])
