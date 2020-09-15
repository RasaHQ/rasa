from typing import Tuple, Text, List

import pytest

from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    RESPONSE,
    INTENT_RESPONSE_KEY,
    ACTION_TEXT,
    ACTION_NAME,
)
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


def test_tokens_comparison():
    x = Token("hello", 0)
    y = Token("Hello", 0)

    assert x == x
    assert y < x

    assert x != 1

    with pytest.raises(TypeError):
        assert y < "a"


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [("Forecast for lunch", ["Forecast", "for", "lunch"], [(0, 8), (9, 12), (13, 18)])],
)
def test_train_tokenizer(
    text: Text, expected_tokens: List[Text], expected_indices: List[Tuple[int]]
):
    tk = WhitespaceTokenizer()

    message = Message.build(text=text)
    message.set(RESPONSE, text)
    message.set(INTENT, text)

    training_data = TrainingData()
    training_data.training_examples = [message]

    tk.train(training_data)

    for attribute in [RESPONSE, TEXT]:
        tokens = training_data.training_examples[0].get(TOKENS_NAMES[attribute])

        assert [t.text for t in tokens] == expected_tokens
        assert [t.start for t in tokens] == [i[0] for i in expected_indices]
        assert [t.end for t in tokens] == [i[1] for i in expected_indices]

    # check intent attribute
    tokens = training_data.training_examples[0].get(TOKENS_NAMES[INTENT])

    assert [t.text for t in tokens] == [text]


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [("Forecast for lunch", ["Forecast", "for", "lunch"], [(0, 8), (9, 12), (13, 18)])],
)
def test_train_tokenizer_e2e_actions(
    text: Text, expected_tokens: List[Text], expected_indices: List[Tuple[int]]
):
    tk = WhitespaceTokenizer()

    message = Message.build(text=text)
    message.set(ACTION_TEXT, text)
    message.set(ACTION_NAME, text)

    training_data = TrainingData()
    training_data.training_examples = [message]

    tk.train(training_data)

    for attribute in [ACTION_TEXT, TEXT]:
        tokens = training_data.training_examples[0].get(TOKENS_NAMES[attribute])

        assert [t.text for t in tokens] == expected_tokens
        assert [t.start for t in tokens] == [i[0] for i in expected_indices]
        assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [("Forecast for lunch", ["Forecast", "for", "lunch"], [(0, 8), (9, 12), (13, 18)])],
)
def test_train_tokenizer_action_name(
    text: Text, expected_tokens: List[Text], expected_indices: List[Tuple[int]]
):
    tk = WhitespaceTokenizer()

    message = Message.build(text=text)
    message.set(ACTION_NAME, text)

    training_data = TrainingData()
    training_data.training_examples = [message]

    tk.train(training_data)

    # check action_name attribute
    tokens = training_data.training_examples[0].get(TOKENS_NAMES[ACTION_NAME])

    assert [t.text for t in tokens] == [text]


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [("Forecast for lunch", ["Forecast", "for", "lunch"], [(0, 8), (9, 12), (13, 18)])],
)
def test_process_tokenizer(
    text: Text, expected_tokens: List[Text], expected_indices: List[Tuple[int]]
):
    tk = WhitespaceTokenizer()

    message = Message.build(text=text)

    tk.process(message)

    tokens = message.get(TOKENS_NAMES[TEXT])

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.parametrize(
    "text, expected_tokens", [("action_listen", ["action", "listen"])]
)
def test_process_tokenizer_action_name(text: Text, expected_tokens: List[Text]):
    tk = WhitespaceTokenizer({"intent_tokenization_flag": True})

    message = Message.build(text=text)
    message.set(ACTION_NAME, text)

    tk.process(message)

    tokens = message.get(TOKENS_NAMES[ACTION_NAME])

    assert [t.text for t in tokens] == expected_tokens


@pytest.mark.parametrize(
    "text, expected_tokens", [("I am hungry", ["I", "am", "hungry"])]
)
def test_process_tokenizer_action_test(text: Text, expected_tokens: List[Text]):
    tk = WhitespaceTokenizer({"intent_tokenization_flag": True})

    message = Message.build(text=text)
    message.set(ACTION_NAME, text)
    message.set(ACTION_TEXT, text)

    tk.process(message)

    tokens = message.get(TOKENS_NAMES[ACTION_TEXT])
    assert [t.text for t in tokens] == expected_tokens

    message.set(ACTION_TEXT, "")
    tk.process(message)
    tokens = message.get(TOKENS_NAMES[ACTION_NAME])
    assert [t.text for t in tokens] == [text]


@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("Forecast_for_LUNCH", ["Forecast_for_LUNCH"]),
        ("Forecast for LUNCH", ["Forecast for LUNCH"]),
    ],
)
def test_split_intent(text: Text, expected_tokens: List[Text]):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = WhitespaceTokenizer(component_config)

    message = Message.build(text=text)
    message.set(INTENT, text)

    assert [t.text for t in tk._split_name(message, INTENT)] == expected_tokens


@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("faq/ask_language", ["faq", "ask_language"]),
        ("faq/ask+language", ["faq", "ask", "language"]),
    ],
)
def test_split_intent_response_key(text, expected_tokens):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = WhitespaceTokenizer(component_config)

    message = Message.build(text=text)
    message.set(INTENT_RESPONSE_KEY, text)

    assert [
        t.text for t in tk._split_name(message, attribute=INTENT_RESPONSE_KEY)
    ] == expected_tokens


@pytest.mark.parametrize(
    "token_pattern, tokens, expected_tokens",
    [
        (
            None,
            [Token("hello", 0), Token("there", 6)],
            [Token("hello", 0), Token("there", 6)],
        ),
        (
            "",
            [Token("hello", 0), Token("there", 6)],
            [Token("hello", 0), Token("there", 6)],
        ),
        (
            r"(?u)\b\w\w+\b",
            [Token("role-based", 0), Token("access-control", 11)],
            [
                Token("role", 0),
                Token("based", 5),
                Token("access", 11),
                Token("control", 18),
            ],
        ),
        (
            r".*",
            [Token("role-based", 0), Token("access-control", 11)],
            [Token("role-based", 0), Token("access-control", 11)],
        ),
        (
            r"(test)",
            [Token("role-based", 0), Token("access-control", 11)],
            [Token("role-based", 0), Token("access-control", 11)],
        ),
    ],
)
def test_apply_token_pattern(
    token_pattern: Text, tokens: List[Token], expected_tokens: List[Token]
):
    component_config = {"token_pattern": token_pattern}

    tokenizer = WhitespaceTokenizer(component_config)
    actual_tokens = tokenizer._apply_token_pattern(tokens)

    assert len(actual_tokens) == len(expected_tokens)
    for actual_token, expected_token in zip(actual_tokens, expected_tokens):
        assert actual_token.text == expected_token.text
        assert actual_token.start == expected_token.start
        assert actual_token.end == expected_token.end


@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("Forecast_for_LUNCH", ["Forecast_for_LUNCH"]),
        ("Forecast for LUNCH", ["Forecast for LUNCH"]),
        ("Forecast+for+LUNCH", ["Forecast", "for", "LUNCH"]),
    ],
)
def test_split_action_name(text: Text, expected_tokens: List[Text]):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = WhitespaceTokenizer(component_config)

    message = Message.build(text=text)
    message.set(ACTION_NAME, text)

    assert [t.text for t in tk._split_name(message, ACTION_NAME)] == expected_tokens
