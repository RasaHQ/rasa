from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import urlparse
import json

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

def create_emulator(service):
    if (service is None):
        from emulators import NoEmulator
        return NoEmulator()
    elif(service.lower() == 'wit'):
        from emulators.wit import WitEmulator
        return WitEmulator()
    elif(service.lower() == 'luis'):
        from emulators.luis import LUISEmulator
        return LUISEmulator()
    else:
        raise ValueError("unknown service : {0}".format(service))



class DataRouter(object):
    def __init__(self,backend=None,service=None):
        self.interpreter = create_interpreter(backend)
        self.emulator = create_emulator(service)

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


try:
    router = DataRouter()
    server = HTTPServer(('', PORT_NUMBER), ParsaRequestHandler)
    print 'Started httpserver on port ' , PORT_NUMBER
    server.serve_forever()

except KeyboardInterrupt:
    print '^C received, shutting down the web server'
    server.socket.close()


