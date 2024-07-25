from unittest.mock import patch, MagicMock, call, Mock

from rasa.llm_fine_tuning.conversation_storage import (
    FileStorageStrategy,
    StorageContext,
)


@patch("os.path.basename")
@patch("rasa.utils.io.write_yaml")
def test_write(mock_write_yaml: Mock, mock_basename: Mock, tmpdir: str):
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
    file_storage.write(mock_conversation, sub_dir)

    # Assert
    mock_basename.assert_called_with("test_file.yml")
    mock_conversation.as_dict.assert_called_once()
    mock_write_yaml.assert_called_with({"key": "value"}, expected_file_path)


@patch.object(FileStorageStrategy, "write")
def test_write_conversation(mock_write: Mock, tmpdir: str):
    # Arrange
    mock_conversation = MagicMock()
    storage_strategy = FileStorageStrategy(tmpdir)
    storage_context = StorageContext(storage_strategy)

    # Act
    storage_context.write_conversation(mock_conversation)

    # Assert
    mock_write.assert_called_once_with(mock_conversation, None)


@patch.object(FileStorageStrategy, "write")
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
