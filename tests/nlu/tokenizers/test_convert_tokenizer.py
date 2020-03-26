import pytest

from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import TEXT, INTENT, TOKENS_NAMES
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "Forecast for lunch",
            ["forecast", "for", "lunch"],
            [(0, 8), (9, 12), (13, 18)],
        ),
        ("hello", ["hello"], [(0, 5)]),
        ("you're", ["you", "re"], [(0, 3), (4, 6)]),
        ("r. n. b.", ["r", "n", "b"], [(0, 1), (3, 4), (6, 7)]),
        ("rock & roll", ["rock", "&", "roll"], [(0, 4), (5, 6), (7, 11)]),
        (
            "ńöñàśçií",
            ["ń", "ö", "ñ", "à", "ś", "ç", "i", "í"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
        ),
    ],
)
def test_convert_tokenizer_edge_cases(text, expected_tokens, expected_indices):
    tk = ConveRTTokenizer()

    tokens = tk.tokenize(Message(text), attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("Forecast_for_LUNCH", ["Forecast_for_LUNCH"]),
        ("Forecast for LUNCH", ["Forecast for LUNCH"]),
    ],
)
def test_custom_intent_symbol(text, expected_tokens):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = ConveRTTokenizer(component_config)

    message = Message(text)
    message.set(INTENT, text)

    tk.train(TrainingData([message]))

    assert [t.text for t in message.get(TOKENS_NAMES[INTENT])] == expected_tokens
