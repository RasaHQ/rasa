from rasa_nlu.tokenizers import Tokenizer


class SpacyTokenizer(Tokenizer):

    def __init__(self, nlp):
        self.nlp = nlp

    def tokenize(self, text):
        return [t.text for t in self.nlp(text)]
