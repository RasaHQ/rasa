from mitie import tokenize
import nltk


class MITIETokenizer(object):
    def __init__(self):
        pass

    def tokenize(self, text):
        return [w.decode('utf-8') for w in nltk.word_tokenize(text.encode('utf-8'))]
