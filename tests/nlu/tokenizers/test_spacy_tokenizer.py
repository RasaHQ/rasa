import pytest

from rasa.nlu.training_data import TrainingData
from rasa.nlu.training_data import Message
from rasa.nlu.constants import (
    CLS_TOKEN,
    TEXT,
    SPACY_DOCS,
    INTENT,
    RESPONSE,
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
    message.set(SPACY_DOCS[TEXT], spacy_nlp(text))

    tokens = tk.tokenize(message, attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.parametrize(
    "text, expected_pos_tags",
    [
        ("Forecast for lunch", ["NN", "IN", "NN"]),
        ("Hello, how are you?", ["UH", ",", "WRB", "VBP", "PRP", "."]),
    ],
)
def test_spacy_pos_tags(text, expected_pos_tags, spacy_nlp):
    tk = SpacyTokenizer()

    message = Message(text)
    message.set(SPACY_DOCS[TEXT], spacy_nlp(text))

    tokens = tk.tokenize(message, attribute=TEXT)

    assert [t.data.get("pos") for t in tokens] == expected_pos_tags


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
    message.set(SPACY_DOCS[TEXT], spacy_nlp(text))
    message.set(RESPONSE, text)
    message.set(SPACY_DOCS[RESPONSE], spacy_nlp(text))

    training_data = TrainingData()
    training_data.training_examples = [message]

    tk.train(training_data)

    for attribute in [RESPONSE, TEXT]:
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
    message.set(SPACY_DOCS[TEXT], spacy_nlp(text))
    message.set(INTENT, text)

    tk.train(TrainingData([message]))

    assert [t.text for t in message.get(TOKENS_NAMES[INTENT])] == expected_tokens
