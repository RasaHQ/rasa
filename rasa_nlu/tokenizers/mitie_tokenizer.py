from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import str
import re

from typing import Any
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

from rasa_nlu.tokenizers import Tokenizer
from rasa_nlu.components import Component


class MitieTokenizer(Tokenizer, Component):
    name = "tokenizer_mitie"

    context_provides = {
        "process": ["tokens"],
    }

    def __init__(self):
        pass

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie"]

    def tokenize(self, text):
        # type: (Text) -> List[Text]
        from mitie import tokenize

        return [w.decode('utf-8') for w in tokenize(text.encode('utf-8'))]

    def process(self, text):
        # type: (Text) -> Dict[Text, Any]

        return {
            "tokens": self.tokenize(text)
        }

    def tokenize_with_offsets(self, text):
        # type: (Text) -> Tuple[List[Text], List[int]]
        from mitie import tokenize

        _text = text.encode('utf-8')
        offsets = []
        offset = 0
        tokens = [w.decode('utf-8') for w in tokenize(_text)]
        for tok in tokens:
            m = re.search(re.escape(tok), text[offset:], re.UNICODE)
            if m is None:
                message = "Invalid MITIE offset. Token '{}' in message '{}'.".format(str(tok),
                                                                                     str(text.encode('utf-8')))
                raise ValueError(message)
            offsets.append(offset + m.start())
            offset += m.end()
        return tokens, offsets
