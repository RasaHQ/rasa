# -*- coding: utf-8 -*-


from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


def test_whitespace():
    from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    tk = WhitespaceTokenizer()
    assert tk.tokenize("Hi. My name is rasa") == ['Hi.', 'My', 'name', 'is', 'rasa']
    assert tk.tokenize("hello ńöñàśçií") == ['hello', 'ńöñàśçií']


def test_spacy(spacy_nlp):

    def tokenize_sentence(sentence, expected_result):
        from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
        tk = SpacyTokenizer()
        assert tk.tokenize(sentence, spacy_nlp) == expected_result

    tokenize_sentence("Hi. My name is rasa", ['Hi', '.', 'My', 'name', 'is', 'rasa'])
    tokenize_sentence("hello ńöñàśçií", ['hello', 'ńöñàśçií'])


def test_mitie():
    from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
    tk = MitieTokenizer()

    assert tk.tokenize("Hi. My name is rasa") == ['Hi', 'My', 'name', 'is', 'rasa']
    assert tk.tokenize("ὦ ἄνδρες ᾿Αθηναῖοι") == ['ὦ', 'ἄνδρες', '᾿Αθηναῖοι']
    assert tk.tokenize_with_offsets("Forecast for lunch") == (['Forecast', 'for', 'lunch'], [0, 9, 13])
    assert tk.tokenize_with_offsets("hey ńöñàśçií how're you?") == (
        ['hey', 'ńöñàśçií', 'how', '\'re', 'you', '?'],
        [0, 4, 13, 16, 20, 23])
