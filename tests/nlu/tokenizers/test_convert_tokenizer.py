import pytest

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES, NUMBER_OF_SUB_TOKENS
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer

# TODO
#   skip tests as the ConveRT model is not publicly available anymore (see https://github.com/RasaHQ/rasa/issues/6806)


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
@pytest.mark.skip
def test_convert_tokenizer_edge_cases(
    component_builder, text, expected_tokens, expected_indices
):
    tk = component_builder.create_component_from_class(ConveRTTokenizer)

    tokens = tk.tokenize(Message(data={TEXT: text}), attribute=TEXT)

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
@pytest.mark.skip
def test_custom_intent_symbol(component_builder, text, expected_tokens):
    tk = component_builder.create_component_from_class(
        ConveRTTokenizer, intent_tokenization_flag=True, intent_split_symbol="+"
    )

    message = Message(data={TEXT: text})
    message.set(INTENT, text)

    tk.train(TrainingData([message]))

    assert [t.text for t in message.get(TOKENS_NAMES[INTENT])] == expected_tokens


@pytest.mark.parametrize(
    "text, expected_number_of_sub_tokens",
    [("Aarhus is a city", [2, 1, 1, 1]), ("sentence embeddings", [1, 3])],
)
@pytest.mark.skip
def test_convert_tokenizer_number_of_sub_tokens(
    component_builder, text, expected_number_of_sub_tokens
):
    tk = component_builder.create_component_from_class(ConveRTTokenizer)

    message = Message(data={TEXT: text})
    message.set(INTENT, text)

    tk.train(TrainingData([message]))

    assert [
        t.get(NUMBER_OF_SUB_TOKENS) for t in message.get(TOKENS_NAMES[TEXT])
    ] == expected_number_of_sub_tokens
