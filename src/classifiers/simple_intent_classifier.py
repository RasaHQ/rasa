from rasa_nlu.components import Component


class SimpleIntentClassifier(Component):

    name = "intent_simple"

    his = ["hello", "hi", "hey"]

    byes = ["bye", "goodbye"]

    def process(self, text):
        return {
            "intent": self.parse(text)
        }

    def parse(self, text):
        _text = text.lower()

        def is_present(x): return x in _text

        if any(map(is_present, self.his)):
            return "greet"
        elif any(map(is_present, self.byes)):
            return "goodbye"
        else:
            return "None"
