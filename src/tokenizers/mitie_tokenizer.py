import logging
from mitie import tokenize
import re


class MITIETokenizer(object):
    def __init__(self):
        pass

    def tokenize(self, text):
        return [w.decode('utf-8') for w in tokenize(text.encode('utf-8'))]

    def tokenize_with_offsets(self, text):
        _text = text.encode('utf-8')
        offsets = []
        offset = 0
        tokens = [w.decode('utf-8') for w in tokenize(_text)]
        for tok in tokens:
            m = re.search(re.escape(tok), text[offset:], re.UNICODE)
            if m is None:
                message = u"Failed to calculate MITIE offset for token '{}' in message'{}' :".format(tok, text)
                logging.error(message)
                raise ValueError(message)
            offsets.append(offset + m.start())
            offset += m.end()
        return tokens, offsets
