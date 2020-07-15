import pytest

from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import TEXT, INTENT, TOKENS_NAMES, NUMBER_OF_SUB_TOKENS
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "forecast for lunch",
            ["forecast", "for", "lunch"],
            [(0, 8), (9, 12), (13, 18)],
        ),
        ("hello", ["hello"], [(0, 5)]),
        ("you're", ["you", "re"], [(0, 3), (4, 6)]),
        ("r. n. b.", ["r", "n", "b"], [(0, 1), (3, 4), (6, 7)]),
        ("rock & roll", ["rock", "&", "roll"], [(0, 4), (5, 6), (7, 11)]),
        ("ńöñàśçií", ["ńöñàśçií"], [(0, 8)]),
    ],
)
def test_convert_tokenizer_edge_cases(
    component_builder, text, expected_tokens, expected_indices
):
    tk = component_builder.create_component_from_class(ConveRTTokenizer)

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
def test_custom_intent_symbol(component_builder, text, expected_tokens):
    tk = component_builder.create_component_from_class(
        ConveRTTokenizer, intent_tokenization_flag=True, intent_split_symbol="+"
    )

    message = Message(text)
    message.set(INTENT, text)

    tk.train(TrainingData([message]))

    assert [t.text for t in message.get(TOKENS_NAMES[INTENT])] == expected_tokens


@pytest.mark.parametrize(
    "text, expected_number_of_sub_tokens",
    [("Aarhus is a city", [2, 1, 1, 1]), ("sentence embeddings", [1, 3])],
)
def test_convert_tokenizer_number_of_sub_tokens(
    component_builder, text, expected_number_of_sub_tokens
):
    tk = component_builder.create_component_from_class(ConveRTTokenizer)

    message = Message(text)
    message.set(INTENT, text)

    tk.train(TrainingData([message]))

    assert [
        t.get(NUMBER_OF_SUB_TOKENS) for t in message.get(TOKENS_NAMES[TEXT])
    ] == expected_number_of_sub_tokens
