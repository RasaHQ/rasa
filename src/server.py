from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
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

    def do_GET(self):
        self._set_headers()
        self.wfile.write('{"result":"ok"}')
        return

    def do_POST(self):
        if self.path=="/parse":
            self._set_headers()
            self.data_string = self.rfile.read(int(self.headers['Content-Length']))            
            data = router.extract(json.loads(self.data_string))            
            result = router.parse(data["text"])
            response = router.format(result)
            self.wfile.write(json.dumps(response))
        return


try:
    router = DataRouter()
    server = HTTPServer(('', PORT_NUMBER), ParsaRequestHandler)
    print 'Started httpserver on port ' , PORT_NUMBER
    server.serve_forever()

except KeyboardInterrupt:
    print '^C received, shutting down the web server'
    server.socket.close()


