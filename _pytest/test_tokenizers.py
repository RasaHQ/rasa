# -*- coding: utf-8 -*-

import pytest


def test_whitespace():
    from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    tk = WhitespaceTokenizer()
    assert tk.tokenize(u"Hi. My name is rasa") == [u'Hi.', u'My', u'name', u'is', u'rasa']
    assert tk.tokenize(u"hello ńöñàśçií") == [u'hello', u'ńöñàśçií']


def test_spacy():
    import spacy

    def tokenize_sentence(sentence, expected_result, language):
        nlp = spacy.load(language, parser=False, entity=False, matcher=False)
        from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
        tk = SpacyTokenizer()
        assert tk.tokenize(sentence, nlp=nlp) == expected_result

    tokenize_sentence(u"Hi. My name is rasa", [u'Hi', u'.', u'My', u'name', u'is', u'rasa'], 'en')
    tokenize_sentence(u"hello ńöñàśçií", [u'hello', u'ńöñàśçií'], 'en')
    tokenize_sentence(u"Hallo. Mein name ist rasa", [u'Hallo', u'.', u'Mein', u'name', u'ist', u'rasa'], 'de')


def test_mitie():
    from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer
    tk = MITIETokenizer()

    tk.tokenize(u"Hi. My name is rasa") == [u'Hi', u'My', u'name', u'is', u'rasa']
    tk.tokenize(u"ὦ ἄνδρες ᾿Αθηναῖοι.") == [u'ὦ', u'ἄνδρες', u'᾿Αθηναῖοι']
    tk.tokenize_with_offsets(u"Forecast for lunch") == ([u'Forecast', u'for', u'lunch'], [0, 9, 13])
