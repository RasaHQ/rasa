from rasa.nlu.tokenizers.hf_transformers_tokenizer import HFTransformersTokenizer

import pytest

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "我想去吃兰州拉面",  # easy/normal case
            ["我", "想", "去", "吃", "兰", "州", "拉", "面"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
        ),
        (
            "从东畈村走了。",  # OOV case: `畈` is a OOV word
            ["从", "东", "[UNK]", "村", "走", "了", "。"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
        ),
        (
            "Micheal 你好吗？",  # Chinese mixed up with English
            ["[UNK]", "你", "好", "吗", "？"],
            [(0, 7), (8, 9,), (9, 10), (10, 11), (11, 12)],
        ),
        (
            "我想买 iPhone 12 🤭",  # Chinese mixed up with English, numbers, and emoji
            ["我", "想", "买", "[UNK]", "12", "[UNK]"],
            [(0, 1), (1, 2), (2, 3), (4, 10), (11, 13), (14, 15)],
        ),
    ],
)
def test_tokenizer_for_chinese(text, expected_tokens, expected_indices):
    tk = HFTransformersTokenizer({"model_weights": "bert-base-chinese"})

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
def test_custom_intent_symbol(text, expected_tokens):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = HFTransformersTokenizer(component_config)

    message = Message(data={TEXT: text})
    message.set(INTENT, text)

    tk.train(TrainingData([message]))

    assert [t.text for t in message.get(TOKENS_NAMES[INTENT])] == expected_tokens
