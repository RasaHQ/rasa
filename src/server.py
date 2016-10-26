from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import urlparse
import json
import argparse

PORT_NUMBER = 5000


def create_interpreter(backend):
    if (backend is None):
        from backends.simple_interpreter import HelloGoodbyeInterpreter
        return HelloGoodbyeInterpreter()
    elif(backend.lower() == 'mitie'):
        from backends.mitie_interpreter import MITIEInterpreter
        return MITIEInterpreter(config.classifier_file,config.ner_file,config.fe_file)
    else:
        raise ValueError("unknown backend : {0}".format(backend))

def create_emulator(mode):
    if (mode is None):
        from emulators import NoEmulator
        return NoEmulator()
    elif(mode.lower() == 'wit'):
        from emulators.wit import WitEmulator
        return WitEmulator()
    elif(mode.lower() == 'luis'):
        from emulators.luis import LUISEmulator
        return LUISEmulator()
    else:
        raise ValueError("unknown mode : {0}".format(mode))



class DataRouter(object):
    def __init__(self,backend=None,mode=None,**kwargs):
        self.interpreter = create_interpreter(backend)
        self.emulator = create_emulator(mode)

    def extract(self,data):
        return self.emulator.normalise_request_json(data)    

    def parse(self,text):
        return self.interpreter.parse(text)

    def format(self,data):
        return self.emulator.normalise_response_json(data)


class ParsaRequestHandler(BaseHTTPRequestHandler):
    
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def get_response(self,data_dict):
        print("response for {0}".format(data_dict))
        data = router.extract(data_dict) 
        print("extracted : {0}".format(data))
        result = router.parse(data["text"])
        response = router.format(result)
        return json.dumps(response)

    def do_GET(self):
        if self.path.startswith("/parse"):
            self._set_headers()
            parsed_path = urlparse.urlparse(self.path)
            data = urlparse.parse_qs(parsed_path.query)
            self.wfile.write(self.get_response(data))
        return

    def do_POST(self):
        if self.path=="/parse":
            self._set_headers()
            data_string = self.rfile.read(int(self.headers['Content-Length']))            
            data_dict = json.loads(data_string)
            self.wfile.write(self.get_response(data_dict))
        return


parser = argparse.ArgumentParser(description='parse incoming text')
parser.add_argument('--backend', default=None, choices=['mitie','sklearn'],help='which backend to use to interpret text (default: None i.e. use built in keyword matcher).')
parser.add_argument('--mode', default=None, choices=['wit','luis'], help='which service to emulate (default: None i.e. use simple built in format)')
parser.add_argument('--port', default=5000, type=int, help='port on which to run server')
args = parser.parse_args()
print(args)
print(vars(args))
try:
    router = DataRouter(**vars(args))
    server = HTTPServer(('', args.port), ParsaRequestHandler)
    print 'Started httpserver on port ' , args.port
    server.serve_forever()

except KeyboardInterrupt:
    print '^C received, shutting down the web server'
    server.socket.close()


