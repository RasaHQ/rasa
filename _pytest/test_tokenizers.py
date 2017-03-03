# -*- coding: utf-8 -*-


def test_whitespace():
    from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    tk = WhitespaceTokenizer()
    assert tk.tokenize(u"Hi. My name is rasa") == [u'Hi.', u'My', u'name', u'is', u'rasa']
    assert tk.tokenize(u"hello ńöñàśçií") == [u'hello', u'ńöñàśçií']


def test_spacy(spacy_nlp_en):
    def tokenize_sentence(sentence, expected_result):
        from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
        tk = SpacyTokenizer(spacy_nlp_en)
        assert tk.tokenize(sentence) == expected_result

    tokenize_sentence(u"Hi. My name is rasa", [u'Hi', u'.', u'My', u'name', u'is', u'rasa'])
    tokenize_sentence(u"hello ńöñàśçií", [u'hello', u'ńöñàśçií'])


def test_mitie():
    from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer
    tk = MITIETokenizer()

    assert tk.tokenize(u"Hi. My name is rasa") == [u'Hi', u'My', u'name', u'is', u'rasa']
    assert tk.tokenize(u"ὦ ἄνδρες ᾿Αθηναῖοι") == [u'ὦ', u'ἄνδρες', u'᾿Αθηναῖοι']
    assert tk.tokenize_with_offsets(u"Forecast for lunch") == ([u'Forecast', u'for', u'lunch'], [0, 9, 13])
    assert tk.tokenize_with_offsets(u"hey ńöñàśçií how're you?") == ([u'hey', u'ńöñàśçií', u'how', u'\'re', 'you', '?'],
                                                                            [0, 4, 13, 16, 20, 23])
