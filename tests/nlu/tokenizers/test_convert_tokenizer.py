import pytest

from rasa.nlu.constants import CLS_TOKEN
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer


def test_convert_tokenizer():
    tk = ConveRTTokenizer()

    assert [t.text for t in tk.tokenize_using_convert("Forecast for lunch")] == [
        "forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]
    assert [t.lemma for t in tk.tokenize_using_convert("Forecast for lunch")] == [
        "forecast",
        "for",
        "lunch",
        CLS_TOKEN,
    ]
    assert [t.start for t in tk.tokenize_using_convert("Forecast for lunch")] == [
        0,
        9,
        13,
        19,
    ]


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        ("hello", ["hello", CLS_TOKEN], [(0, 5), (6, 13)]),
        ("you're", ["you", "re", CLS_TOKEN], [(0, 3), (4, 6), (7, 14)]),
        ("r. n. b.", ["r", "n", "b", CLS_TOKEN], [(0, 1), (3, 4), (6, 7), (8, 15)]),
        ("rock & roll", ["rock", "roll", CLS_TOKEN], [(0, 4), (7, 11), (12, 19)]),
        (
            "ńöñàśçií",
            ["ń", "ö", "ñ", "à", "ś", "ç", "i", "í", CLS_TOKEN],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (9, 16)],
        ),
    ],
)
def test_convert_tokenizer_edge_cases(text, expected_tokens, expected_indices):
    tk = ConveRTTokenizer()

    tokens = tk.tokenize_using_convert(text)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]
