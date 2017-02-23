import re

from mitie import tokenize

from rasa_nlu.tokenizers import Tokenizer


class MITIETokenizer(Tokenizer):
    def __init__(self):
        pass

    def tokenize(self, text, nlp=None):
        return [w.decode('utf-8') for w in tokenize(text.encode('utf-8'))]

    def tokenize_with_offsets(self, text):
        _text = text.encode('utf-8')
        offsets = []
        offset = 0
        tokens = [w.decode('utf-8') for w in tokenize(_text)]
        for tok in tokens:
            m = re.search(re.escape(tok), text[offset:], re.UNICODE)
            offsets.append(offset + m.start())
            offset += m.end()
        return tokens, offsets
