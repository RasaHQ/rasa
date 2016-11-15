

def test_whitespace():
    from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    tk =  WhitespaceTokenizer()
    str = "Hi. My name is rasa"
    assert tk.tokenize(str) == ['Hi.', 'My', 'name', 'is', 'rasa']

def test_spacy():
    from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
    tk =  SpacyTokenizer()
    str = u"Hi. My name is rasa"
    assert tk.tokenize(str) == [u'Hi',u'.', u'My', u'name', u'is', u'rasa']

def test_mitie():
    from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer
    tk =  MITIETokenizer()
    str = u"Hi. My name is rasa"
    assert tk.tokenize(str) == ['Hi', 'My', 'name', 'is', 'rasa']

