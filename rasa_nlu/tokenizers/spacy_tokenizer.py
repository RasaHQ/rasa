from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.tokenizers import Tokenizer
from rasa_nlu.components import Component


class SpacyTokenizer(Tokenizer, Component):
    name = "tokenizer_spacy"

    context_provides = {
        "process": ["tokens"],
    }

    def process(self, text, spacy_nlp):
        # type: (Text, Language) -> Dict[Text, Any]
        from spacy.language import Language

        return {
            "tokens": self.tokenize(text, spacy_nlp)
        }

    def tokenize(self, text, spacy_nlp):
        # type: (Text, Language) -> List[Text]
        from spacy.language import Language

        return [t.text for t in spacy_nlp(text)]
