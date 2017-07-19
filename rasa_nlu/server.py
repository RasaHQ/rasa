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
from twisted.internet.defer import inlineCallbacks, returnValue, maybeDeferred

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
    """Class representing Rasa NLU http server"""

    app = Klein()

    def __init__(self, config, component_builder=None, testing=False):
        logging.basicConfig(filename=config['log_file'], level=config['log_level'])
        logging.captureWarnings(True)
        logger.debug("Configuration: " + config.view())

        logger.debug("Creating a new data router")
        self.config = config
        self.data_router = DataRouter(config, component_builder)
        self.testing = testing
        reactor.suggestThreadPoolSize(config['num_threads'] * 5)

    @app.route("/", methods=['GET'])
    def hello(self, request):
        """Main Rasa route to check if the server is online"""

        return "hello from Rasa NLU: " + __version__

    @app.route("/parse", methods=['GET', 'POST'])
    @requires_auth
    @inlineCallbacks
    def parse_get(self, request):
        request.setHeader('Content-Type', 'application/json')
        if request.method.decode('utf-8', 'strict') == 'GET':
            request_params = {key.decode('utf-8', 'strict'): value[0].decode('utf-8', 'strict') for key, value in
                              request.args.items()}
        else:
            request_params = json.loads(request.content.read().decode('utf-8', 'strict'))
        if 'q' not in request_params:
            request.setResponseCode(404)
            returnValue(json.dumps({"error": "Invalid parse parameter specified"}))
        else:
            data = self.data_router.extract(request_params)
            try:
                request.setResponseCode(200)
                response = yield (maybeDeferred(self.data_router.parse, data)
                                  if self.testing
                                  else threads.deferToThread(self.data_router.parse, data))
                returnValue(json.dumps(response))
            except InvalidModelError as e:
                request.setResponseCode(404)
                returnValue(json.dumps({"error": "{}".format(e)}))

    @app.route("/version", methods=['GET'])
    @requires_auth
    def version(self, request):
        """Returns the Rasa server's version"""

        request.setHeader('Content-Type', 'application/json')
        return json.dumps({'version': __version__})

    @app.route("/config", methods=['GET'])
    @requires_auth
    def rasaconfig(self, request):
        """Returns the in-memory configuration of the Rasa server"""

        request.setHeader('Content-Type', 'application/json')
        return json.dumps(self.config.as_dict())

    @app.route("/status", methods=['GET'])
    @requires_auth
    def status(self, request):
        request.setHeader('Content-Type', 'application/json')
        return json.dumps(self.data_router.get_status())

    @app.route("/train", methods=['POST'])
    @requires_auth
    def train(self, request):
        def errback(f):
            f.trap(ValueError)
            logger.debug("error: {}".format(f.getErrorMessage()))

        data_string = request.content.read().decode('utf-8', 'strict')
        kwargs = {key.decode('utf-8', 'strict'): value[0].decode('utf-8', 'strict') for key, value in
                  request.args.items()}
        request.setHeader('Content-Type', 'application/json')
        request.setResponseCode(200)

        async_training = self.data_router.start_train_process(data_string, kwargs)
        async_training.addErrback(errback)

        return json.dumps({"info": "training started."})


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argparser()
    cmdline_args = {key: val for key, val in list(vars(arg_parser.parse_args()).items()) if val is not None}
    rasa_nlu_config = RasaNLUConfig(cmdline_args.get("config"), os.environ, cmdline_args)
    rasa = RasaNLU(rasa_nlu_config)
    logger.info('Started http server on port %s' % rasa_nlu_config['port'])
    # rasa.app.run('0.0.0.0', rasa_nlu_config['port'])

    from treq.testing import StubTreq

    stub = StubTreq(rasa.app.resource())

    training_data = '{"rasa_nlu_data":{"entity_examples":[{"text":"hey","intent":"greet","entities":[]},{"text":"howdy","intent":"greet","entities":[]},{"text":"hey there","intent":"greet","entities":[]},{"text":"hello","intent":"greet","entities":[]},{"text":"hi","intent":"greet","entities":[]},{"text":"i\'m looking for a place to eat","intent":"restaurant_search","entities":[]},{"text":"i\'m looking for a place in the north of town","intent":"restaurant_search","entities":[]},{"text":"show me chinese restaurants","intent":"restaurant_search","entities":[]},{"text":"yes","intent":"affirm","entities":[]},{"text":"yep","intent":"affirm","entities":[]},{"text":"yeah","intent":"affirm","entities":[]},{"text":"show me a mexican place in the centre","intent":"restaurant_search","entities":[]},{"text":"bye","intent":"goodbye","entities":[]},{"text":"goodbye","intent":"goodbye","entities":[]},{"text":"good bye","intent":"goodbye","entities":[]},{"text":"stop","intent":"goodbye","entities":[]},{"text":"end","intent":"goodbye","entities":[]},{"text":"i am looking for an indian spot","intent":"restaurant_search","entities":[]},{"text":"search for restaurants","intent":"restaurant_search","entities":[]},{"text":"anywhere in the west","intent":"restaurant_search","entities":[]},{"text":"central indian restaurant","intent":"restaurant_search","entities":[]},{"text":"indeed","intent":"affirm","entities":[]},{"text":"that\'s right","intent":"affirm","entities":[]},{"text":"ok","intent":"affirm","entities":[]},{"text":"great","intent":"affirm","entities":[]}],"intent_examples":[{"text":"hey","intent":"greet"},{"text":"howdy","intent":"greet"},{"text":"hey there","intent":"greet"},{"text":"hello","intent":"greet"},{"text":"hi","intent":"greet"},{"text":"i\'m looking for a place to eat","intent":"restaurant_search"},{"text":"i\'m looking for a place in the north of town","intent":"restaurant_search"},{"text":"show me chinese restaurants","intent":"restaurant_search"},{"text":"yes","intent":"affirm"},{"text":"yep","intent":"affirm"},{"text":"yeah","intent":"affirm"},{"text":"show me a mexican place in the centre","intent":"restaurant_search"},{"text":"bye","intent":"goodbye"},{"text":"goodbye","intent":"goodbye"},{"text":"good bye","intent":"goodbye"},{"text":"stop","intent":"goodbye"},{"text":"end","intent":"goodbye"},{"text":"i am looking for an indian spot","intent":"restaurant_search"},{"text":"search for restaurants","intent":"restaurant_search"},{"text":"anywhere in the west","intent":"restaurant_search"},{"text":"central indian restaurant","intent":"restaurant_search"},{"text":"indeed","intent":"affirm"},{"text":"that\'s right","intent":"affirm"},{"text":"ok","intent":"affirm"},{"text":"great","intent":"affirm"}]}}'


    @inlineCallbacks
    def test_get():
        response = yield stub.get('http://localhost/parse', params={'q': 'hello'},
                                   data=training_data)
        result = yield response.text()
        print(result)


    test_get()
