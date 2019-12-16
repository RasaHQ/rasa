import pytest

from rasa.nlu.constants import CLS_TOKEN
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer


def test_convert_tokenizer_cls_token():
    component_config = {"use_cls_token": True}

    tk = ConveRTTokenizer(component_config)

    assert [t.text for t in tk.tokenize_using_convert("Forecast for lunch")] == [
        "﹏forecast",
        "﹏for",
        "﹏lunch",
        CLS_TOKEN,
    ]
    assert [t.start for t in tk.tokenize_using_convert("Forecast for lunch")] == [
        0,
        9,
        13,
        19,
    ]


def test_convert_tokenizer():
    component_config = {"use_cls_token": False}

    tk = ConveRTTokenizer(component_config)

    assert [t.text for t in tk.tokenize_using_convert("Forecast for lunch")] == [
        "﹏forecast",
        "﹏for",
        "﹏lunch",
    ]
    assert [t.lemma for t in tk.tokenize_using_convert("Forecast for lunch")] == [
        "﹏forecast",
        "﹏for",
        "﹏lunch",
    ]
    assert [t.start for t in tk.tokenize_using_convert("Forecast for lunch")] == [
        0,
        9,
        13,
    ]


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        ("hello", ["﹏hello"], [(0, 5)]),
        ("you're", ["﹏you", "﹏re"], [(0, 3), (4, 6)]),
        ("r. n. b.", ["﹏r", "﹏n", "﹏b"], [(0, 1), (3, 4), (6, 7)]),
        ("rock & roll", ["﹏rock", "﹏roll"], [(0, 4), (7, 11)]),
        (
            "ńöñàśçií",
            ["﹏", "ń", "ö", "ñ", "à", "ś", "ç", "i", "í"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
        ),
    ],
)
def test_convert_tokenizer_edge_cases(text, expected_tokens, expected_indices):
    component_config = {"use_cls_token": False}
    tk = ConveRTTokenizer(component_config)

    tokens = tk.tokenize_using_convert(text)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]
