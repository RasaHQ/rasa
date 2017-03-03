from rasa_nlu import Interpreter


class HelloGoodbyeInterpreter(Interpreter):
    def __init__(self):
        self.name = "hello_goodbye"
        self.his = ["hello", "hi", "hey"]
        self.byes = ["bye", "goodbye"]

    def parse(self, text):
        _text = text.lower()

        def is_present(x): return x in _text

        if any(map(is_present, self.his)):
            intent = "greet"
        elif any(map(is_present, self.byes)):
            intent = "goodbye"
        else:
            intent = "None"

        return {'text': text, 'intent': intent, 'entities': [], 'confidence': 1.0}
