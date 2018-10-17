from rasa_nlu.model import Interpreter
import json

interpreter = Interpreter.load("./models/current/df-agent")
message = u'Show me details of the 2017 kia rio'

result = interpreter.parse(message)
print(json.dumps(result, indent=2))