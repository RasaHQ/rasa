from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from rasa_nlu.visualization import create_html
from rasa_nlu.training_data import TrainingData
import sys


class NLUVisualizationServer(object):
    def __init__(self, config):
        self.server = None
        self.config = config

    def start(self):
        self.server = HTTPServer(('', 8080), lambda *args: VisualizationRequestHandler(*args))
        print('Started http server at http://0.0.0.0:8080')
        self.server.serve_forever()

    def stop(self):
        print '^C received. Aborting.'
        if self.server is not None:
            print 'shutting down server'
            self.server.socket.close()


class VisualizationRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args):
        BaseHTTPRequestHandler.__init__(self, *args)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        data_file = sys.argv[1]
        training_data = TrainingData(data_file, 'mitie', 'en')
        data = create_html(training_data)
        self.wfile.write(data)
        return


if __name__ == "__main__":

    try:
        server = NLUVisualizationServer(None)
        server.start()
    except KeyboardInterrupt:
        server.stop()
