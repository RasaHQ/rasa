from rasa.nlu.constants import CLS_TOKEN
from rasa.nlu import training_data
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer


def test_spacy(spacy_nlp):
    tk = SpacyTokenizer()

    text = "Forecast for lunch"
    assert [t.text for t in tk.tokenize(spacy_nlp(text))] == [
        "Forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]
    assert [t.lemma for t in tk.tokenize(spacy_nlp(text))] == [
        "forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]

    assert [t.start for t in tk.tokenize(spacy_nlp(text))] == [0, 9, 13, 19]

    text = "hey ńöñàśçií how're you?"
    assert [t.text for t in tk.tokenize(spacy_nlp(text))] == [
        "hey",
        "ńöñàśçií",
        "how",
        "'re",
        "you",
        "?",
        CLS_TOKEN,
    ]
    assert [t.start for t in tk.tokenize(spacy_nlp(text))] == [0, 4, 13, 16, 20, 23, 25]


def test_spacy_intent_tokenizer(spacy_nlp_component):

    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    spacy_nlp_component.train(td, config=None)
    spacy_tokenizer = SpacyTokenizer()
    spacy_tokenizer.train(td, config=None)

    intent_tokens_exist = [
        True if example.get("intent_tokens") is not None else False
        for example in td.intent_examples
    ]

    # no intent tokens should have been set
    assert not any(intent_tokens_exist)
