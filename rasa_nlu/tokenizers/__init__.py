from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object


def tokenizer_from_name(name, language):
    from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
    from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer

    if name == MitieTokenizer.name:
        return MitieTokenizer()
    elif name == SpacyTokenizer.name:
        import spacy
        nlp = spacy.load(language, parser=False, entity=False)
        return SpacyTokenizer(nlp)
    elif name == WhitespaceTokenizer.name:
        return WhitespaceTokenizer()


class Tokenizer(object):
    def tokenize(self, text):
        # type: (str) -> [str]
        raise NotImplementedError
