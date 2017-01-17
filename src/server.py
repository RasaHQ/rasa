import argparse
import json
import os
import urlparse
import multiprocessing
import glob
import warnings
import logging
import signal
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from rasa_nlu.train import do_train
from rasa_nlu.config import RasaNLUConfig


class RasaNLUServer(object):
    def __init__(self, config):
        self.server = None
        self.config = config
        self.logfile = config.write
        self.emulator = self.__create_emulator()
        self.interpreter = self.__create_interpreter()
        self.data_router = DataRouter(config, self.interpreter, self.emulator)

        if 'DYNO' in os.environ and config.backend == 'mitie':  # running on Heroku
            from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
            MITIEFeaturizer(config.mitie_file)

    def __create_interpreter(self):
        model_dir = self.config.server_model_dir
        metadata, backend = None, None

        if model_dir is not None:
            # download model from S3 if needed
            if not os.path.isdir(model_dir):
                try:
                    from rasa_nlu.persistor import Persistor
                    p = Persistor(self.config.path, self.config.aws_region, self.config.bucket_name)
                    p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
                except:
                    warnings.warn("using default interpreter, couldn't find model dir or fetch it from S3")

            metadata = json.loads(open(os.path.join(model_dir, 'metadata.json'), 'rb').read())
            backend = metadata["backend"]

        if backend is None:
            from interpreters.simple_interpreter import HelloGoodbyeInterpreter
            return HelloGoodbyeInterpreter()
        elif backend.lower() == 'mitie':
            logging.info("using mitie backend")
            from interpreters.mitie_interpreter import MITIEInterpreter
            return MITIEInterpreter(**metadata)
        elif backend.lower() == 'spacy_sklearn':
            logging.info("using spacy + sklearn backend")
            from interpreters.spacy_sklearn_interpreter import SpacySklearnInterpreter
            return SpacySklearnInterpreter(**metadata)
        else:
            raise ValueError("unknown backend : {0}".format(backend))

    def __create_emulator(self):
        mode = self.config.emulate
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

    def start(self):
        self.server = HTTPServer(('', self.config.port), lambda *args: RasaRequestHandler(self.data_router, *args))
        logging.info('Started http server on port %s' % self.config.port)
        self.server.serve_forever()

    def stop(self):
        logging.info('^C received. Aborting.')
        if len(self.data_router.responses) > 0:
            logging.info('saving logs')
            self.data_router.write_logs()
        if self.server is not None:
            logging.info('shutting down server')
            self.server.socket.close()


class DataRouter(object):
    def __init__(self, config, interpreter, emulator):
        self.config = config
        self.interpreter = interpreter
        self.emulator = emulator
        self.logfile = config.write
        self.responses = set()
        self.train_proc = None
        self.model_dir = config.path
        self.token = config.token

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
        logging.info("starting train")
        if self.train_proc is not None and self.train_proc.is_alive():
            self.train_proc.terminate()
            logging.info("training process {0} killed".format(self.train_proc))

        fname = 'tmp_training_data.json'
        with open(fname, 'w') as f:
            f.write(data)
        _config = dict(self.config.items())
        _config["data"] = fname
        train_config = RasaNLUConfig(cmdline_args=_config)

        self.train_proc = multiprocessing.Process(target=do_train, args=(train_config,))
        self.train_proc.start()
        logging.info("training process {0} started".format(self.train_proc))


class RasaRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, data_router, *args):
        self.data_router = data_router
        BaseHTTPRequestHandler.__init__(self, *args)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def auth_err(self):
        self.send_response(401)
        self.wfile.write("unauthorized")

    def get_response(self, data_dict):
        if u'q' not in data_dict:
            return json.dumps({"error": "Invalid parse parameter specified"})

        data = self.data_router.extract(data_dict)
        result = self.data_router.parse(data["text"])
        response = self.data_router.format(result)
        return json.dumps(response)

    def do_GET(self):
        if self.data_router.auth(self.path):
            self._set_headers()
            if self.path.startswith("/parse"):
                parsed_path = urlparse.urlparse(urlparse.unquote(self.path).decode('utf-8'))
                data = urlparse.parse_qs(parsed_path.query)
                self.wfile.write(self.get_response(data))
            elif self.path.startswith("/status"):
                response = self.data_router.get_status()
                self.wfile.write(response)
            else:
                self.wfile.write("hello")
        else:
            self.auth_err()
        return

    def do_POST(self):
        if self.data_router.auth(self.path):
            if self.path.startswith("/parse"):
                self._set_headers()
                data_string = self.rfile.read(int(self.headers['Content-Length']))
                data_dict = json.loads(data_string.decode("utf-8"))
                self.wfile.write(self.get_response(data_dict))

            if self.path.startswith("/train"):
                self._set_headers()
                data_string = self.rfile.read(int(self.headers['Content-Length']))
                self.data_router.start_train_proc(data_string)
                self.data_router.start_train_proc(data_string.decode("utf-8"))
                self.wfile.write(
                    json.dumps({"info": "training started with pid {0}".format(self.data_router.train_proc.pid)})
                )
        else:
            self.auth_err()
        return


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


if __name__ == "__main__":
    parser = create_argparser()
    cmdline_args = {key: val for key, val in vars(parser.parse_args()).items() if val is not None}
    config = RasaNLUConfig(cmdline_args.get("config"), os.environ, cmdline_args)
    print(config.view())
    logging.basicConfig(filename=config.log_file, level=config.log_level)
    logging.captureWarnings(True)
    logging.debug(config.view())
    try:
        def stop(signal_number, frame):
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, stop)
        server = RasaNLUServer(config)
        server.start()

    except KeyboardInterrupt:
        server.stop()
