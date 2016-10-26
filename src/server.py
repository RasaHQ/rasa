from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import json

PORT_NUMBER = 5000



def create_interpreter(backend):
    if (backend is None):
        return HelloGoodbyeInterpreter()
    elif(backend.lower() == 'mitie'):
        from mitie_interpreter import MITIEInterpreter
        return MITIEInterpreter(config.classifier_file,config.ner_file,config.fe_file)
    else:
        raise ValueError("Unknown backend type")

class ParsingRouter(object):
    def __init__(self,backend=None):
        self.interpreter = create_interpreter(backend)

    def parse(self,text):
        return self.interpreter.parse(text)


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
            data = json.loads(self.data_string)
            result = router.parse(data["text"])
            s= json.dumps(result)
            self.wfile.write(s)
        return


try:
    router = ParsingRouter()
    server = HTTPServer(('', PORT_NUMBER), ParsaRequestHandler)
    print(vars(server))
    print 'Started httpserver on port ' , PORT_NUMBER
    server.serve_forever()

except KeyboardInterrupt:
    print '^C received, shutting down the web server'
    server.socket.close()


