from rasa_nlu import Interpreter


class HelloGoodbyeInterpreter(Interpreter):
    def __init__(self):
        self.name="hello_goodbye"
        self.his = ["hello","hi","hey"]
        self.byes = ["bye","goodbye"]

    def parse(self,text):
        _text, intent = text.lower(), "None"
        is_present = lambda x: x in _text

        if (True in map(is_present,self.his)):
            intent="greet"
        elif (True in map(is_present,self.byes)):
            intent="goodbye"

        return {'text':text,'intent':intent,'entities':[]}
