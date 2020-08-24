import pytest

from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import TEXT, INTENT, TOKENS_NAMES
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "Forecast for lunch",
            ["Forecast", "for", "lunch"],
            [(0, 8), (9, 12), (13, 18)],
        ),
        (
            "hey ńöñàśçií how're you?",
            ["hey", "ńöñàśçií", "how", "'re", "you", "?"],
            [(0, 3), (4, 12), (13, 16), (16, 19), (20, 23), (23, 24)],
        ),
    ],
)
def test_mitie(text, expected_tokens, expected_indices):
    tk = MitieTokenizer()

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

    tk = MitieTokenizer(component_config)

    message = Message(text)
    message.set(INTENT, text)

    tk.train(TrainingData([message]))

    assert [t.text for t in message.get(TOKENS_NAMES[INTENT])] == expected_tokens
