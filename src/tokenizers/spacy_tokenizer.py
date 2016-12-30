import spacy


class SpacyTokenizer(object):

    def __init__(self):
        pass

    def tokenize(self, text, nlp=None):
        return [t.text for t in nlp(text)]
