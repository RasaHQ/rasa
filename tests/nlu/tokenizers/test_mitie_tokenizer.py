from typing import Dict, Text, Any, Tuple, List, Callable

import pytest

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer


@pytest.fixture()
def tokenizer(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
) -> Callable[[Dict[Text, Any]], MitieTokenizer]:
    def inner(config: Dict[Text, Any]) -> MitieTokenizer:
        return MitieTokenizer.create(
            {**MitieTokenizer.get_default_config(), **config},
            default_model_storage,
            Resource("mitie_tokenizer"),
            default_execution_context,
        )

    return inner


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
def test_mitie(
    text: Text,
    expected_tokens: List[Text],
    expected_indices: List[Tuple[int, int]],
    tokenizer: Callable[[Dict[Text, Any]], MitieTokenizer],
):
    tk = tokenizer({})

    tokens = tk.tokenize(Message.build(text=text), attribute=TEXT)

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
def test_custom_intent_symbol(
    text: Text,
    expected_tokens: List[Text],
    tokenizer: Callable[[Dict[Text, Any]], MitieTokenizer],
):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = tokenizer(component_config)

    message = Message.build(text=text)
    message.set(INTENT, text)

    tk.process_training_data(TrainingData([message]))

    assert [t.text for t in message.get(TOKENS_NAMES[INTENT])] == expected_tokens
