import pytest

from rasa.nlu.constants import TOKENS_NAMES, TEXT_ATTRIBUTE, INTENT_ATTRIBUTE
from rasa.nlu.training_data import TrainingData, Message
from tests.nlu import utilities
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


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
            ["hey", "ńöñàśçií", "how", "re", "you"],
            [(0, 3), (4, 12), (13, 16), (17, 19), (20, 23)],
        ),
        (
            "https://www.google.com/search?client=safari&rls=en&q=i+like+rasa&ie=UTF-8&oe=UTF-8 https://rasa.com/docs/nlu/components/#tokenizer-whitespace",
            [
                "https://www.google.com/search?"
                "client=safari&rls=en&q=i+like+rasa&ie=UTF-8&oe=UTF-8",
                "https://rasa.com/docs/nlu/components/#tokenizer-whitespace",
            ],
            [(0, 82), (83, 141)],
        ),
    ],
)
def test_whitespace(text, expected_tokens, expected_indices):

    tk = WhitespaceTokenizer()

    tokens = tk.tokenize(Message(text), attribute=TEXT_ATTRIBUTE)

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

    tk = WhitespaceTokenizer(component_config)

    message = Message(text)
    message.set(INTENT_ATTRIBUTE, text)

    tk.train(TrainingData([message]))

    assert [
        t.text for t in message.get(TOKENS_NAMES[INTENT_ATTRIBUTE])
    ] == expected_tokens


@pytest.mark.parametrize(
    "text, component_config, expected_tokens",
    [
        ("Forecast for LUNCH", {}, ["Forecast", "for", "LUNCH"]),
        ("Forecast for LUNCH", {"case_sensitive": False}, ["forecast", "for", "lunch"]),
        ("Forecast for LUNCH", {"case_sensitive": True}, ["Forecast", "for", "LUNCH"]),
    ],
)
def test_whitespace_with_case(text, component_config, expected_tokens):

    tk = WhitespaceTokenizer(component_config)

    message = Message(text)

    tokens = tk.tokenize(message, attribute=TEXT_ATTRIBUTE)

    assert [t.text for t in tokens] == expected_tokens


def test_whitespace_training():
    _config = utilities.base_test_conf("supervised_embeddings")

    examples = [
        Message(
            "Any Mexican restaurant will do",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 4, "end": 11, "value": "Mexican", "entity": "cuisine"}
                ],
            },
        ),
        Message(
            "I want Tacos!",
            {
                "intent": "restaurant_search",
                "entities": [
                    {"start": 7, "end": 12, "value": "Mexican", "entity": "cuisine"}
                ],
            },
        ),
    ]

    component_config = {"case_sensitive": False}
    tk = WhitespaceTokenizer(component_config)

    tk.train(TrainingData(training_examples=examples), _config)

    assert examples[0].data.get("tokens")[0].text == "any"
    assert examples[0].data.get("tokens")[1].text == "mexican"
    assert examples[0].data.get("tokens")[2].text == "restaurant"
    assert examples[0].data.get("tokens")[3].text == "will"
    assert examples[0].data.get("tokens")[4].text == "do"
    assert examples[1].data.get("tokens")[0].text == "i"
    assert examples[1].data.get("tokens")[1].text == "want"
    assert examples[1].data.get("tokens")[2].text == "tacos"
