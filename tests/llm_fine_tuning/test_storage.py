from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

import rasa.shared.utils.io
from rasa.cli.e2e_test import read_test_cases
from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.llm_fine_tuning.llm_data_preparation_module import LLMDataExample
from rasa.llm_fine_tuning.storage import FileStorageStrategy, StorageContext
from rasa.llm_fine_tuning.train_test_split_module import (
    ConversationalDataFormat,
    ConversationalMessageDataFormat,
    InstructionDataFormat,
)


@patch("os.path.basename")
@patch("rasa.utils.io.write_yaml")
def test_write_conversation(mock_write_yaml: Mock, mock_basename: Mock, tmpdir: str):
    # Arrange
    mock_basename.return_value = "test_file.yml"
    mock_conversation = MagicMock()
    mock_conversation.original_e2e_test_case.file = "test_file.yml"
    mock_conversation.as_dict.return_value = {"key": "value"}
    output_dir = tmpdir
    sub_dir = "sub"
    file_storage = FileStorageStrategy(output_dir)

    expected_file_path = f"{output_dir}/{sub_dir}/test_file.yml"

    # Act
    file_storage.write_conversation(mock_conversation, sub_dir)

    # Assert
    mock_basename.assert_called_with("test_file.yml")
    mock_conversation.as_dict.assert_called_once()
    mock_write_yaml.assert_called_with({"key": "value"}, Path(expected_file_path))


@patch.object(FileStorageStrategy, "write_conversation")
def test_write_conversation_without_storage_location(mock_write: Mock, tmpdir: str):
    # Arrange
    mock_conversation = MagicMock()
    storage_strategy = FileStorageStrategy(tmpdir)
    storage_context = StorageContext(storage_strategy)

    # Act
    storage_context.write_conversation(mock_conversation)

    # Assert
    mock_write.assert_called_once_with(mock_conversation, None)


@patch.object(FileStorageStrategy, "write_conversation")
def test_write_conversations(mock_write: Mock, tmpdir: str):
    # Arrange
    mock_conversations = [MagicMock(), MagicMock(), MagicMock()]
    storage_strategy = FileStorageStrategy(tmpdir)
    storage_context = StorageContext(storage_strategy)

    # Act
    storage_context.write_conversations(mock_conversations)

    # Assert
    calls = [call(conversation, None) for conversation in mock_conversations]
    mock_write.assert_has_calls(calls, any_order=False)


def test_write_llm_data(tmpdir: str):
    # Arrange
    llm_data = [
        LLMDataExample(
            "prompt",
            [StartFlowCommand("flow1")],
            "original test name",
            "original user utterance",
            "rephrased user utterance",
        )
    ]
    output_dir = tmpdir
    storage_location = "3_llm_finetune_data/llm_ft_data.jsonl"
    file_storage = FileStorageStrategy(output_dir)

    expected_file_path = f"{output_dir}/{storage_location}"

    # Act
    file_storage.write_llm_data(llm_data, storage_location)

    # Assert
    loaded_data = rasa.shared.utils.io.read_json_file(expected_file_path)

    assert loaded_data == llm_data[0].as_dict()


def test_write_llm_data_writes_unicode(tmpdir: str):
    # Arrange
    umlauts = "ä, ö, ü"
    llm_data = [
        LLMDataExample(
            "prompt",
            [StartFlowCommand("flow1")],
            f"original test name with umlauts {umlauts}",
            f"original user utterance with umlauts {umlauts}",
            f"rephrased user utterance with umlauts {umlauts}",
        )
    ]
    output_dir = tmpdir
    storage_location = "3_llm_finetune_data/llm_ft_data.jsonl"
    file_storage = FileStorageStrategy(output_dir)

    expected_file_path = f"{output_dir}/{storage_location}"

    # Act
    file_storage.write_llm_data(llm_data, storage_location)

    # Assert
    loaded_data = rasa.shared.utils.io.read_json_file(expected_file_path)

    assert loaded_data == llm_data[0].as_dict()

    with open(expected_file_path, encoding="utf-8") as f:
        text = f.read()

    # The characters themselves must be there …
    assert "ä" in text and "ö" in text and "ü" in text
    # … and the escaped versions must NOT be there.
    assert r"\u00e4" not in text and r"\u00f6" not in text and r"\u00fc" not in text


def test_write_formatted_finetuning_data_alpaca_format(tmpdir: str):
    # Arrange
    formatted_data = [
        InstructionDataFormat("value1", "value2"),
        InstructionDataFormat("value3", "value4"),
    ]
    output_dir = tmpdir
    storage_location = "4_train_test_split"
    file_name = "ft_splits/train.jsonl"
    file_storage = FileStorageStrategy(output_dir)

    expected_file_path = f"{output_dir}/{storage_location}/{file_name}"

    # Act
    file_storage.write_formatted_finetuning_data(
        formatted_data, storage_location, file_name
    )

    # Assert
    loaded_data = rasa.shared.utils.io.read_jsonl_file(expected_file_path)
    expected_data = [example.as_dict() for example in formatted_data]

    assert loaded_data == expected_data


def test_write_formatted_finetuning_data_sharegpt_format(tmpdir: str):
    # Arrange
    formatted_data = [
        ConversationalDataFormat(
            [
                ConversationalMessageDataFormat("role1", "text1"),
                ConversationalMessageDataFormat("role2", "text2"),
                ConversationalMessageDataFormat("role3", "text3"),
            ]
        )
    ]

    output_dir = tmpdir
    storage_location = "4_train_test_split"
    file_name = "ft_splits/train.jsonl"
    file_storage = FileStorageStrategy(output_dir)

    expected_file_path = f"{output_dir}/{storage_location}/{file_name}"

    # Act
    file_storage.write_formatted_finetuning_data(
        formatted_data, storage_location, file_name
    )

    # Assert
    loaded_data = rasa.shared.utils.io.read_jsonl_file(expected_file_path)
    expected_data = [example.as_dict() for example in formatted_data]

    assert loaded_data == expected_data


def test_write_formatted_finetuning_data_writes_unicode(tmp_path):
    # -----------------------------------------------------------------------
    # Arrange
    umlauts = "ä, ö, ü"
    formatted_data = [
        ConversationalDataFormat(
            [
                ConversationalMessageDataFormat("system", f"intro {umlauts}"),
                ConversationalMessageDataFormat("user", f"question {umlauts}"),
                ConversationalMessageDataFormat("assistant", "answer"),
            ]
        )
    ]

    storage_location = "4_train_test_split"
    file_name = "ft_splits/train.jsonl"

    # FileStorageStrategy is assumed to create directories as needed
    file_storage = FileStorageStrategy(output_dir=str(tmp_path))

    expected_file_path = tmp_path / storage_location / file_name

    # -----------------------------------------------------------------------
    # Act
    file_storage.write_formatted_finetuning_data(
        formatted_data, storage_location, file_name
    )

    # -----------------------------------------------------------------------
    # Assert 1 – JSON round-trip works
    loaded_jsonl = rasa.shared.utils.io.read_jsonl_file(str(expected_file_path))
    expected_jsonl = [ex.as_dict() for ex in formatted_data]
    assert loaded_jsonl == expected_jsonl  # still identical objects

    # -----------------------------------------------------------------------
    # Assert 2 – file really contains readable UTF-8 characters
    raw_text = expected_file_path.read_text(encoding="utf-8")

    # characters themselves must be there …
    assert "ä" in raw_text and "ö" in raw_text and "ü" in raw_text
    # … and the escaped ASCII versions must not be there.
    assert r"\u00e4" not in raw_text
    assert r"\u00f6" not in raw_text
    assert r"\u00fc" not in raw_text


@pytest.mark.parametrize(
    "file_name",
    [
        "data/test_llm_finetuning/e2e_test_sample_with_metadata.yml",
        "data/test_llm_finetuning/e2e_test_sample_with_fixtures.yml",
        "data/test_llm_finetuning/e2e_test_sample.yml",
    ],
)
def test_write_e2e_test_suite_to_yaml_file(tmpdir: str, file_name: str):
    # Read the test suite.
    test_suite = read_test_cases(file_name)

    output_dir = tmpdir
    storage_location = "4_train_test_split"
    file_name = "e2e_tests/train.yaml"
    file_storage = FileStorageStrategy(output_dir)

    expected_file_path = f"{output_dir}/{storage_location}/{file_name}"

    # Write the test suite to a file.
    file_storage.write_e2e_test_suite_to_yaml_file(
        test_suite, storage_location, file_name
    )

    # Read the test suite from the newly created file.
    loaded_data = read_test_cases(expected_file_path)

    # Expected matches original except file_name becomes the path we wrote to.
    def _norm_paths(obj: object, old: str, new: str) -> object:
        if isinstance(obj, dict):
            return {
                k: (new if k == "file_name" and v == old else _norm_paths(v, old, new))
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_norm_paths(x, old, new) for x in obj]
        return obj

    source = (
        test_suite.test_cases[0].file
        if test_suite.test_cases
        else test_suite.fixtures_per_test[0].file
    )
    assert loaded_data.as_dict() == _norm_paths(
        test_suite.as_dict(), source, expected_file_path
    )
