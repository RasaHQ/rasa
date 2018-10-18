from rasa_nlu.model import Interpreter
import json

interpreter = Interpreter.load("./models/current/df-agent")
message = u'I will like some rice and chicken'

result = interpreter.parse(message)
print(json.dumps(result, indent=2))