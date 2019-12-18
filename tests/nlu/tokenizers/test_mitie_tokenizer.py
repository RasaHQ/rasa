from rasa.nlu.constants import CLS_TOKEN
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer


def test_mitie():
    tk = MitieTokenizer()

    text = "Forecast for lunch"
    assert [t.text for t in tk.tokenize(text)] == [
        "Forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]
    assert [t.start for t in tk.tokenize(text)] == [0, 9, 13, 19]

    text = "hey ńöñàśçií how're you?"
    assert [t.text for t in tk.tokenize(text)] == [
        "hey",
        "ńöñàśçií",
        "how",
        "'re",
        "you",
        "?",
        CLS_TOKEN,
    ]
    assert [t.start for t in tk.tokenize(text)] == [0, 4, 13, 16, 20, 23, 25]
