import spacy

class SpacyTokenizer(object):
    def __init__(self):
        self.nlp = spacy.load('en',tagger=False, parser=False, entity=False, matcher=False,add_vectors=False)
    def tokenize(self,text):
        return [t.text.encode('utf-8') for t in self.nlp(text)]
