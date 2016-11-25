from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import urlparse, json, argparse, os
from rasa_nlu.util import update_config


def create_interpreter(config):

    model_dir = config.get("server_model_dir")
    backend = None
    if (model_dir is not None):
        metadata = json.loads(open(os.path.join(model_dir,'metadata.json'),'rb').read())
        backend = metadata["backend"]

    if (backend is None):
        from interpreters.simple_interpreter import HelloGoodbyeInterpreter
        return HelloGoodbyeInterpreter()
    elif(backend.lower() == 'mitie'):
        print("using mitie backend")
        from interpreters.mitie_interpreter import MITIEInterpreter
        return MITIEInterpreter(**metadata)
    elif(backend.lower() == 'spacy_sklearn'):
        print("using spacy + sklearn backend")
        from interpreters.spacy_sklearn_interpreter import SpacySklearnInterpreter
        return SpacySklearnInterpreter(**metadata)
    else:
        raise ValueError("unknown backend : {0}".format(backend))

def create_emulator(config):
    mode = config.get('emulate')
    if (mode is None):
        from emulators import NoEmulator
        return NoEmulator()
    elif(mode.lower() == 'wit'):
        from emulators.wit import WitEmulator
        return WitEmulator()
    elif(mode.lower() == 'luis'):
        from emulators.luis import LUISEmulator
        return LUISEmulator()
    elif(mode.lower() == 'api'):
        from emulators.api import ApiEmulator
        return ApiEmulator()
    else:
        raise ValueError("unknown mode : {0}".format(mode))

def create_argparser():
    parser = argparse.ArgumentParser(description='parse incoming text')
    parser.add_argument('-d','--server_model_dir', default=None, help='directory where model files are saved')
    parser.add_argument('-e','--emulate', default=None, choices=['wit','luis', 'api'], help='which service to emulate (default: None i.e. use simple built in format)')
    parser.add_argument('-P','--port', default=5000, type=int, help='port on which to run server')
    parser.add_argument('-c','--config', default=None, help="config file, all the command line options can also be passed via a (json-formatted) config file. NB command line args take precedence")
    parser.add_argument('-w','--write', default='rasa_nlu_log.json', help='file where logs will be saved')
    parser.add_argument('-l', '--language', default='en', choices=['de', 'en'], help="model and data language")

    return parser

class DataRouter(object):
    def __init__(self,**config):
        self.interpreter = create_interpreter(config)
        self.emulator = create_emulator(config)
        self.logfile=config["logfile"]
        self.responses = set()

    def extract(self,data):
        return self.emulator.normalise_request_json(data)

    def parse(self,text):
        result = self.interpreter.parse(text)
        self.responses.add(json.dumps(result,sort_keys=True))
        return result

    def format(self,data):
        return self.emulator.normalise_response_json(data)

    def write_logs(self):
        with open(self.logfile,'w') as f:
            responses = [json.loads(r) for r in self.responses]
            f.write(json.dumps(responses,indent=2))

class RasaRequestHandler(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def get_response(self,data_dict):
        data = router.extract(data_dict)
        result = router.parse(data["text"])
        response = router.format(result)
        return json.dumps(response)

    def do_GET(self):
        if self.path.startswith("/parse"):
            self._set_headers()
            parsed_path = urlparse.urlparse(self.path)
            data = urlparse.parse_qs(parsed_path.query)
            self.wfile.write(self.get_response(data))
        else:
            self._set_headers()
            self.wfile.write("hello")
        return

    def do_POST(self):
        if self.path=="/parse":
            self._set_headers()
            data_string = self.rfile.read(int(self.headers['Content-Length']))
            data_dict = json.loads(data_string)
            self.wfile.write(self.get_response(data_dict))
        return

def init():
    parser = create_argparser()
    args = parser.parse_args()
    config = {'logfile': os.path.join(os.getcwd(), 'rasa_nlu_logs.json')} if args.config is None else json.loads(open(args.config,'rb').read())
    config = update_config(config,args,exclude=['config'])
    return config


try:
    config = init()
    print(config)
    router = DataRouter(**config)
    server = HTTPServer(('', config["port"]), RasaRequestHandler)
    print 'Started httpserver on port ' , config["port"]
    server.serve_forever()

except KeyboardInterrupt:
    print '^C received, saving logs'
    router.write_logs()
    print 'shutting down server'
    server.socket.close()
