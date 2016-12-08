from mitie import tokenize


class MITIETokenizer(object):
    def __init__(self):
        pass

    def tokenize(self, text):
        return [w.decode('utf-8') for w in tokenize(text.encode('utf-8'))]
