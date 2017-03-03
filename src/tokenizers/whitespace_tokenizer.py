from rasa_nlu.tokenizers import Tokenizer


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split()
