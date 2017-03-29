from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from rasa_nlu.tokenizers import Tokenizer
from rasa_nlu.components import Component


class SpacyTokenizer(Tokenizer, Component):
    name = "tokenizer_spacy"

    context_provides = {
        "process": ["tokens"],
    }

    def __init__(self, nlp=None):
        self.nlp = nlp

    def pipeline_init(self, spacy_nlp):
        # type: (Language) -> None
        from spacy.language import Language

        self.nlp = spacy_nlp

    def process(self, text):
        # type: (str) -> dict

        return {
            "tokens": self.tokenize(text)
        }

    def tokenize(self, text):
        # type: (str) -> [str]

        return [t.text for t in self.nlp(text)]
