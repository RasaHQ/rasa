class WhitespaceTokenizer(object):
    def tokenize(self, text, nlp=None):
        return text.split()
