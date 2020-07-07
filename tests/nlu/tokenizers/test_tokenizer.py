import pytest

from rasa.nlu.constants import TEXT, INTENT, RESPONSE, MESSAGE_ACTION_NAME, TOKENS_NAMES
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


def test_tokens_comparison():
    from rasa.nlu.tokenizers.tokenizer import Token

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
def test_train_tokenizer(text, expected_tokens, expected_indices):
    tk = WhitespaceTokenizer()

    message = Message(text)
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
    [("Forecast_for_lunch", ["Forecast", "for", "lunch"], [(0, 8), (9, 12), (13, 18)])],
)
def test_train_tokenizer_action_name(text, expected_tokens, expected_indices):
    tk = WhitespaceTokenizer()

    message = Message("")
    message.set(MESSAGE_ACTION_NAME, text)

    training_data = TrainingData()
    training_data.training_examples = [message]

    tk.train(training_data)

    # check that nothing was added to text
    tokens = training_data.training_examples[0].get(TOKENS_NAMES[TEXT])
    assert [t.text for t in tokens] == [""]

    # check that action name was tokenizer correctly in training
    tokens = training_data.training_examples[0].get(TOKENS_NAMES[MESSAGE_ACTION_NAME])
    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [("Forecast for lunch", ["Forecast", "for", "lunch"], [(0, 8), (9, 12), (13, 18)])],
)
def test_process_tokenizer(text, expected_tokens, expected_indices):
    tk = WhitespaceTokenizer()

    message = Message(text)

    tk.process(message)

    tokens = message.get(TOKENS_NAMES[TEXT])

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
def test_split_intent(text, expected_tokens):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = WhitespaceTokenizer(component_config)

    message = Message(text)
    message.set(INTENT, text)

    assert [t.text for t in tk._split_intent(message)] == expected_tokens

@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("Forecast_for_LUNCH", ["Forecast_for_LUNCH"]),
        ("Forecast for LUNCH", ["Forecast for LUNCH"]),
    ],
)
def test_process_with_intent_attribute(text, expected_tokens):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = WhitespaceTokenizer(component_config)

    message = Message(text)
    message.set(INTENT, text)

    tk.process(message, INTENT)
    tokens = message.get(TOKENS_NAMES[INTENT])
    assert [t.text for t in tokens] == expected_tokens

@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("Forecast_for_LUNCH", ["Forecast", "for", "LUNCH"]),
        ("Forecast for LUNCH", ["Forecast for LUNCH"]),
    ],
)
def test_process_with_action_attribute(text, expected_tokens):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "_"}

    tk = WhitespaceTokenizer(component_config)

    message = Message("")
    message.set(MESSAGE_ACTION_NAME, text)

    tk.process(message, MESSAGE_ACTION_NAME)
    tokens = message.get(TOKENS_NAMES[MESSAGE_ACTION_NAME])
    assert [t.text for t in tokens] == expected_tokens


@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("Forecast for LUNCH", ["Forecast", "for", "LUNCH"]),
    ],
)
def test_process_with_action_attribute_and_text(text, expected_tokens):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "_"}

    tk = WhitespaceTokenizer(component_config)

    message = Message(text)
    message.set(MESSAGE_ACTION_NAME, text)

    tk.process(message, MESSAGE_ACTION_NAME)
    tokens = message.get(TOKENS_NAMES[TEXT])
    assert [t.text for t in tokens] == expected_tokens
