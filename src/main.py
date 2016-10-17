from flask import Flask, request

from mitie_interpreter import MITIEInterpreter
import config
import json

interpreter = MITIEInterpreter(config.classifier_file,config.ner_file,config.fe_file)

app = Flask(__name__)


@app.route("/parse", methods = ['GET', 'POST'])
def parse():
    if request.method == 'GET':
        return "ok"
    if request.method == 'POST':
        body = request.json
        try:
            return json.dumps(interpreter.parse(body['text']))
        except:
            if (config.debug_mode):
                raise
                return "error"
            else:
                return "error"

if __name__ == "__main__":
    app.run(port=config.self_port, debug=config.debug_mode)
