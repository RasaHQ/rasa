# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def test_whitespace():
    from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    tk = WhitespaceTokenizer()

    assert [t.text for t in tk.tokenize("Forecast for lunch")] == ['Forecast', 'for', 'lunch']
    assert [t.offset for t in tk.tokenize("Forecast for lunch")] == [0, 9, 13]

    assert [t.text for t in tk.tokenize("hey ńöñàśçií how're you?")] == ['hey', 'ńöñàśçií', 'how\'re', 'you?']
    assert [t.offset for t in tk.tokenize("hey ńöñàśçií how're you?")] == [0, 4, 13, 20]


def test_spacy(spacy_nlp):
    from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer

    tk = SpacyTokenizer()
    assert [t.text for t in tk.tokenize(spacy_nlp("Forecast for lunch"))] == ['Forecast', 'for', 'lunch']
    assert [t.offset for t in tk.tokenize(spacy_nlp("Forecast for lunch"))] == [0, 9, 13]

    assert [t.text for t in tk.tokenize(spacy_nlp("hey ńöñàśçií how're you?"))] == \
           ['hey', 'ńöñàśçií', 'how', '\'re', 'you', '?']
    assert [t.offset for t in tk.tokenize(spacy_nlp("hey ńöñàśçií how're you?"))] == [0, 4, 13, 16, 20, 23]


def test_mitie():
    from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
    tk = MitieTokenizer()

    assert [t.text for t in tk.tokenize("Forecast for lunch")] == ['Forecast', 'for', 'lunch']
    assert [t.offset for t in tk.tokenize("Forecast for lunch")] == [0, 9, 13]

    assert [t.text for t in tk.tokenize("hey ńöñàśçií how're you?")] == ['hey', 'ńöñàśçií', 'how', '\'re', 'you', '?']
    assert [t.offset for t in tk.tokenize("hey ńöñàśçií how're you?")] == [0, 4, 13, 16, 20, 23]
