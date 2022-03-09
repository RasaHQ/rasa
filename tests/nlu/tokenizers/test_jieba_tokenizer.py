import logging
from typing import Dict, Optional

from _pytest.logging import LogCaptureFixture
from _pytest.tmpdir import TempPathFactory

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer

import pytest

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT


def create_jieba(config: Optional[Dict] = None) -> JiebaTokenizer:
    config = config if config else {}
    return JiebaTokenizer.create(
        {**JiebaTokenizer.get_default_config(), **config}, None, None, None
    )


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "我想去吃兰州拉面",
            ["我", "想", "去", "吃", "兰州", "拉面"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8)],
        ),
        (
            "Micheal你好吗？",
            ["Micheal", "你好", "吗", "？"],
            [(0, 7), (7, 9), (9, 10), (10, 11)],
        ),
    ],
)
def test_jieba(text, expected_tokens, expected_indices):
    tk = create_jieba()

    tokens = tk.tokenize(Message(data={TEXT: text}), attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


def test_jieba_load_and_persist_dictionary(
    tmp_path_factory: TempPathFactory,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    caplog: LogCaptureFixture,
):
    dictionary_directory = tmp_path_factory.mktemp("dictionaries")
    dictionary_path = dictionary_directory / "dictionary_1"

    dictionary_contents = """
创新办 3 i
云计算 5
凱特琳 nz
台中
        """
    dictionary_path.write_text(dictionary_contents, encoding="utf-8")

    component_config = {"dictionary_path": dictionary_directory}

    resource = Resource("jieba")
    tk = JiebaTokenizer.create(
        {**JiebaTokenizer.get_default_config(), **component_config},
        default_model_storage,
        resource,
        default_execution_context,
    )

    tk.process_training_data(TrainingData([Message(data={TEXT: ""})]))

    # The dictionary has not been persisted yet.
    with caplog.at_level(logging.DEBUG):
        JiebaTokenizer.load(
            {**JiebaTokenizer.get_default_config(), **component_config},
            default_model_storage,
            resource,
            default_execution_context,
        )
        assert any(
            "Failed to load JiebaTokenizer from model storage." in message
            for message in caplog.messages
        )

    tk.persist()

    # Check the persisted dictionary matches the original file.
    with default_model_storage.read_from(resource) as resource_dir:
        contents = (resource_dir / "dictionary_1").read_text(encoding="utf-8")
        assert contents == dictionary_contents

    # Delete original files to show that we read from the model storage.
    dictionary_path.unlink()
    dictionary_directory.rmdir()

    JiebaTokenizer.load(
        {**JiebaTokenizer.get_default_config(), **component_config},
        default_model_storage,
        resource,
        default_execution_context,
    )

    tk.process([Message(data={TEXT: ""})])


@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("Forecast_for_LUNCH", ["Forecast_for_LUNCH"]),
        ("Forecast for LUNCH", ["Forecast for LUNCH"]),
    ],
)
def test_custom_intent_symbol(text, expected_tokens):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    tk = create_jieba(component_config)

    message = Message(data={TEXT: text})
    message.set(INTENT, text)

    tk.process_training_data(TrainingData([message]))

    assert [t.text for t in message.get(TOKENS_NAMES[INTENT])] == expected_tokens
