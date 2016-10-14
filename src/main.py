from flask import Flask, request

from mitie_interpreter import MITIEInterpreter
import config

interpreter = MITIEInterpreter(config.classifier_file,config.ner_file,config.fe_file)
assert interpreter is not None

app = Flask(__name__)


@app.route("/parse", methods = ['GET', 'POST'])
def parse():
    if request.method == 'GET':
        return "ok"
    if request.method == 'POST':
        body = request.json
        return str(interpreter.parse(body['text']))
        try:
            return interpreter.parse(body['text'])
        except:
            return "oops"

if __name__ == "__main__":
    app.run(port=config.self_port, debug=config.debug_mode)
