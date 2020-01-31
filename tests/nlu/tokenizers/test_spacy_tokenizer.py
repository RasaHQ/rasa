import pytest

from rasa.nlu.training_data import TrainingData
from rasa.nlu.training_data import Message
from rasa.nlu.constants import (
    CLS_TOKEN,
    TEXT_ATTRIBUTE,
    SPACY_DOCS,
    INTENT_ATTRIBUTE,
    RESPONSE_ATTRIBUTE,
    TOKENS_NAMES,
)
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer


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
def test_spacy(text, expected_tokens, expected_indices, spacy_nlp):
    tk = SpacyTokenizer()

    message = Message(text)
    message.set(SPACY_DOCS[TEXT_ATTRIBUTE], spacy_nlp(text))

    tokens = tk.tokenize(message, attribute=TEXT_ATTRIBUTE)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "Forecast for lunch",
            ["Forecast", "for", "lunch", CLS_TOKEN],
            [(0, 8), (9, 12), (13, 18), (19, 26)],
        )
    ],
)
def test_train_tokenizer(text, expected_tokens, expected_indices, spacy_nlp):
    tk = SpacyTokenizer()

    message = Message(text)
    message.set(SPACY_DOCS[TEXT_ATTRIBUTE], spacy_nlp(text))
    message.set(RESPONSE_ATTRIBUTE, text)
    message.set(SPACY_DOCS[RESPONSE_ATTRIBUTE], spacy_nlp(text))

    training_data = TrainingData()
    training_data.training_examples = [message]

    tk.train(training_data)

    for attribute in [RESPONSE_ATTRIBUTE, TEXT_ATTRIBUTE]:
        tokens = training_data.training_examples[0].get(TOKENS_NAMES[attribute])

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
def test_custom_intent_symbol(text, expected_tokens, spacy_nlp):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = SpacyTokenizer(component_config)

    message = Message(text)
    message.set(SPACY_DOCS[TEXT_ATTRIBUTE], spacy_nlp(text))
    message.set(INTENT_ATTRIBUTE, text)

    tk.train(TrainingData([message]))

    assert [
        t.text for t in message.get(TOKENS_NAMES[INTENT_ATTRIBUTE])
    ] == expected_tokens
