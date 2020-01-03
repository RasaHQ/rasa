from rasa.nlu.constants import CLS_TOKEN
from rasa.nlu import training_data


def test_spacy(spacy_nlp):
    from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer

    component_config = {"use_cls_token": False}

    tk = SpacyTokenizer(component_config)

    text = "Forecast for lunch"
    assert [t.text for t in tk.tokenize(spacy_nlp(text))] == [
        "Forecast",
        "for",
        "lunch",
    ]
    assert [t.lemma for t in tk.tokenize(spacy_nlp(text))] == [
        "forecast",
        "for",
        "lunch",
    ]

    assert [t.offset for t in tk.tokenize(spacy_nlp(text))] == [0, 9, 13]

    text = "hey ńöñàśçií how're you?"
    assert [t.text for t in tk.tokenize(spacy_nlp(text))] == [
        "hey",
        "ńöñàśçií",
        "how",
        "'re",
        "you",
        "?",
    ]
    assert [t.offset for t in tk.tokenize(spacy_nlp(text))] == [0, 4, 13, 16, 20, 23]


def test_spacy_add_cls_token(spacy_nlp):
    from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer

    component_config = {"use_cls_token": True}

    tk = SpacyTokenizer(component_config)

    text = "Forecast for lunch"
    assert [t.text for t in tk.tokenize(spacy_nlp(text))] == [
        "Forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]
    assert [t.offset for t in tk.tokenize(spacy_nlp(text))] == [0, 9, 13, 19]


def test_spacy_intent_tokenizer(spacy_nlp_component):
    from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer

    component_config = {"use_cls_token": False}

    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    spacy_nlp_component.train(td, config=None)
    spacy_tokenizer = SpacyTokenizer(component_config)
    spacy_tokenizer.train(td, config=None)

    intent_tokens_exist = [
        True if example.get("intent_tokens") is not None else False
        for example in td.intent_examples
    ]

    # no intent tokens should have been set
    assert not any(intent_tokens_exist)
