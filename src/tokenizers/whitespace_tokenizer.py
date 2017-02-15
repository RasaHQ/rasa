from rasa_nlu.tokenizers import Tokenizer


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text, nlp=None):
        return text.split()
