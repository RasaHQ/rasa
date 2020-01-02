import pytest

from rasa.nlu.training_data import Message
from rasa.nlu.constants import CLS_TOKEN, TEXT_ATTRIBUTE, SPACY_DOCS
from rasa.nlu import training_data
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

    text = "Forecast for lunch"
    message = Message(text)
    message.set(SPACY_DOCS[TEXT_ATTRIBUTE], spacy_nlp(text))

    tokens = tk.tokenize(message, attribute=TEXT_ATTRIBUTE)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


def test_spacy_intent_tokenizer(spacy_nlp_component):

    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    spacy_nlp_component.train(td, config=None)
    spacy_tokenizer = SpacyTokenizer()
    spacy_tokenizer.train(td, config=None)

    intent_tokens_exist = [
        True if example.get("intent_tokens") is not None else False
        for example in td.intent_examples
    ]

    # no intent tokens should have been set
    assert not any(intent_tokens_exist)
