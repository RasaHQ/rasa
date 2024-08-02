from pathlib import Path
from unittest.mock import patch, MagicMock, call, Mock

import rasa.shared.utils.io
from rasa.llm_fine_tuning.llm_data_preparation_module import LLMDataExample
from rasa.llm_fine_tuning.storage import (
    FileStorageStrategy,
    StorageContext,
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
            "start_flow",
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
