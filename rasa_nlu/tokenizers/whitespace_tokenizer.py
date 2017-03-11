from rasa_nlu.tokenizers import Tokenizer
from rasa_nlu.components import Component


class WhitespaceTokenizer(Tokenizer, Component):
    name = "tokenizer_whitespace"

    context_provides = ["tokens"]

    def process(self, text):
        # type: (str) -> dict

        return {
            "tokens": self.tokenize(text)
        }

    def tokenize(self, text):
        # type: (str) -> [str]

        return text.split()
