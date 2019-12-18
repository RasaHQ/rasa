from rasa.nlu.constants import CLS_TOKEN


def test_mitie():
    from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer

    component_config = {"use_cls_token": False}

    tk = MitieTokenizer(component_config)

    text = "Forecast for lunch"
    assert [t.text for t in tk.tokenize(text)] == ["Forecast", "for", "lunch"]
    assert [t.offset for t in tk.tokenize(text)] == [0, 9, 13]

    text = "hey ńöñàśçií how're you?"
    assert [t.text for t in tk.tokenize(text)] == [
        "hey",
        "ńöñàśçií",
        "how",
        "'re",
        "you",
        "?",
    ]
    assert [t.offset for t in tk.tokenize(text)] == [0, 4, 13, 16, 20, 23]


def test_mitie_add_cls_token():
    from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer

    component_config = {"use_cls_token": True}

    tk = MitieTokenizer(component_config)

    text = "Forecast for lunch"
    assert [t.text for t in tk.tokenize(text)] == [
        "Forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]
    assert [t.offset for t in tk.tokenize(text)] == [0, 9, 13, 19]
