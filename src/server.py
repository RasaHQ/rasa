import argparse
import json
import os
import urlparse
import multiprocessing
import glob
import warnings
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

from rasa_nlu.util import update_config
from rasa_nlu.train import do_train


def create_interpreter(config):
    model_dir = config.get("server_model_dir")
    metadata, backend = None, None

    if model_dir is not None:
        # download model from S3 if needed
        if not os.path.isdir(model_dir):
            try:
                from rasa_nlu.persistor import Persistor
                p = Persistor(config['path'], config['aws_region'], config['bucket_name'])
                p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
            except:
                warnings.warn("using default interpreter, couldn't find model dir or fetch it from S3")

        metadata = json.loads(open(os.path.join(model_dir, 'metadata.json'), 'rb').read())
        backend = metadata["backend"]

    if backend is None:
        from interpreters.simple_interpreter import HelloGoodbyeInterpreter
        return HelloGoodbyeInterpreter()
    elif backend.lower() == 'mitie':
        print("using mitie backend")
        from interpreters.mitie_interpreter import MITIEInterpreter
        return MITIEInterpreter(**metadata)
    elif backend.lower() == 'spacy_sklearn':
        print("using spacy + sklearn backend")
        from interpreters.spacy_sklearn_interpreter import SpacySklearnInterpreter
        return SpacySklearnInterpreter(**metadata)
    else:
        raise ValueError("unknown backend : {0}".format(backend))


def create_emulator(config):
    mode = config.get('emulate')
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


def create_argparser():
    parser = argparse.ArgumentParser(description='parse incoming text')
    parser.add_argument('-d', '--server_model_dir', default=None, 
                        help='directory containing model to for parser to use')
    parser.add_argument('-e', '--emulate', default=None, choices=['wit', 'luis', 'api'],
                        help='which service to emulate (default: None i.e. use simple built in format)')
    parser.add_argument('-p', '--path', default=None, help="path where model files will be saved")
    parser.add_argument('-P', '--port', default=5000, type=int, help='port on which to run server')
    parser.add_argument('-c', '--config', default=None,
                        help="config file, all the command line options can also be passed via a (json-formatted) " +
                             "config file. NB command line args take precedence")
    parser.add_argument('-w', '--write', default='rasa_nlu_log.json', help='file where logs will be saved')
    parser.add_argument('-l', '--language', default='en', choices=['de', 'en'], help="model and data language")
    parser.add_argument('-t', '--token', default=None,
                        help="auth token. If set, reject requests which don't provide this token as a query parameter")

    return parser


class DataRouter(object):
    def __init__(self, config):
        self.config = config
        self.interpreter = create_interpreter(config)
        self.emulator = create_emulator(config)
        self.logfile = config["logfile"]
        self.responses = set()
        self.train_proc = None
        self.model_dir = config["path"]
        self.token = config.get("token")

    def extract(self, data):
        return self.emulator.normalise_request_json(data)

    def parse(self, text):
        result = self.interpreter.parse(text)
        self.responses.add(json.dumps(result, sort_keys=True))
        return result

    def format(self, data):
        return self.emulator.normalise_response_json(data)

    def write_logs(self):
        with open(self.logfile, 'w') as f:
            responses = [json.loads(r) for r in self.responses]
            f.write(json.dumps(responses, indent=2))

    def get_status(self):
        if self.train_proc is not None:
            training = self.train_proc.is_alive()
        else:
            training = False

        models = glob.glob(os.path.join(self.model_dir, 'model*'))
        return json.dumps({
          "training": training,
          "available_models": models
        })

    def auth(self, path):

        if self.token is None:
            return True
        else:
            parsed_path = urlparse.urlparse(path)
            data = urlparse.parse_qs(parsed_path.query)
            valid = ("token" in data and data["token"][0] == self.token)
            return valid

    def start_train_proc(self, data):
        print("starting train")
        if self.train_proc is not None and self.train_proc.is_alive():
            self.train_proc.terminate()
            print("training process {0} killed".format(self.train_proc))

        fname = 'tmp_training_data.json'
        with open(fname, 'w') as f:
            f.write(data)
        train_config = self.config.copy()
        train_config["data"] = fname
        self.train_proc = multiprocessing.Process(target=do_train, args=(train_config,))
        self.train_proc.start()
        print("training process {0} started".format(self.train_proc))


class RasaRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def auth_err(self):
        self.send_response(401)
        self.wfile.write("unauthorized")

    def get_response(self, data_dict):
        data = router.extract(data_dict)
        result = router.parse(data["text"])
        response = router.format(result)
        return json.dumps(response)

    def do_GET(self):
        if router.auth(self.path):
            self._set_headers()
            if self.path.startswith("/parse"):
                parsed_path = urlparse.urlparse(self.path)
                data = urlparse.parse_qs(parsed_path.query)
                self.wfile.write(self.get_response(data))
            elif self.path.startswith("/status"):
                response = router.get_status()
                self.wfile.write(response)
            else:
                self.wfile.write("hello")
        else:
            self.auth_err()
        return

    def do_POST(self):
        if router.auth(self.path):
            if self.path.startswith("/parse"):
                self._set_headers()
                data_string = self.rfile.read(int(self.headers['Content-Length']))
                data_dict = json.loads(data_string)
                self.wfile.write(self.get_response(data_dict))

            if self.path.startswith("/train"):
                self._set_headers()
                data_string = self.rfile.read(int(self.headers['Content-Length']))
                router.start_train_proc(data_string)
                self.wfile.write('training started with pid {0}'.format(router.train_proc.pid))
        else:
            self.auth_err()
        return


def init():
    parser = create_argparser()
    args = parser.parse_args()
    config = {'logfile': os.path.join(os.getcwd(), 'rasa_nlu_logs.json')} if args.config is None else json.loads(
        open(args.config, 'rb').read())
    config = update_config(config, args, exclude=['config'])
    return config

router = None
server = None

try:
    config = init()
    router = DataRouter(config)
    server = HTTPServer(('', config["port"]), RasaRequestHandler)
    print 'Started http server on port ', config["port"]
    server.serve_forever()

except KeyboardInterrupt:
    print '^C received. Aborting.'
    if router is not None:
        print 'saving router logs'
        router.write_logs()
    if server is not None:
        print 'shutting down server'
        server.socket.close()
