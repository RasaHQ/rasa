import textwrap
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock

import pandas as pd
import pytest

from rasa.e2e_test.e2e_test_converter import (
    E2ETestConverter,
    ConversationEntry,
    Conversation,
    Conversations,
)
from rasa.shared.exceptions import RasaException

SAMPLE_CONVERSATIONS_CSV_PATH = "data/test_data_convert_e2e/sample_conversations.csv"
SAMPLE_CONVERSATIONS_XLSX_PATH = "data/test_data_convert_e2e/sample_conversations.xlsx"
SAMPLE_CONVERSATION_XLSX_SHEET_NAME = "Sheet1"
UNSUPPORTED_EXTENSION_PATH = "sample_conversations.txt"
UNSUPPORTED_DIRECTORY_PATH = "/data/sample_conversations/"
NUMBER_OF_SAMPLE_CONVERSATIONS = 20

DATA_FRAME_INPUT_DATA = {
    "column1": ["value1", "value2"],
    "column2": ["value3", "value4"],
}
DATA_FRAME_OUTPUT_DATA = [
    {"column1": "value1", "column2": "value3"},
    {"column1": "value2", "column2": "value4"},
]
DATA_FRAME_INPUT_DATA_EMPTY_VALUES = {
    "column1": ["value1", None],
    "column2": [None, "value2"],
}
DATA_FRAME_OUTPUT_DATA_EMPTY_VALUES = [
    {"column1": "value1", "column2": ""},
    {"column1": "", "column2": "value2"},
]

DUMMY_CONVERSATION_1 = {"column1": "user: Hello", "column2": "bot: Hi"}
DUMMY_CONVERSATION_2 = {"column1": "user: How are you?", "column2": "bot: I am fine"}
DUMMY_CONVERSATION_3 = {"column1": "user: Goodbye", "column2": "bot: Bye"}

DUMMY_YAML_TEST = """
- test_case: user_greeting_the_assistant
  steps:
  - user: "Hi"
    assertions:
      - bot_uttered:
          text_matches: "Hello"
""".strip()


@pytest.fixture(autouse=True)
def data_frame() -> pd.DataFrame:
    return pd.DataFrame(DATA_FRAME_INPUT_DATA)


@pytest.fixture(autouse=True)
def data_frame_empty_values() -> pd.DataFrame:
    return pd.DataFrame(DATA_FRAME_INPUT_DATA_EMPTY_VALUES)


@pytest.fixture
def sample_converter() -> E2ETestConverter:
    return E2ETestConverter(path=SAMPLE_CONVERSATIONS_CSV_PATH)


def test_conversation_entry():
    entry = ConversationEntry(data=DUMMY_CONVERSATION_1)
    assert entry.as_dict() == DUMMY_CONVERSATION_1


def test_conversation():
    entries = [
        ConversationEntry(data=DUMMY_CONVERSATION_1),
        ConversationEntry(data=DUMMY_CONVERSATION_2),
    ]
    conversation = Conversation(entries=entries)
    expected_list = [DUMMY_CONVERSATION_1, DUMMY_CONVERSATION_2]
    assert conversation.as_list() == expected_list


def test_conversations():
    entries1 = [
        ConversationEntry(data=DUMMY_CONVERSATION_1),
        ConversationEntry(data=DUMMY_CONVERSATION_2),
    ]
    entries2 = [
        ConversationEntry(data=DUMMY_CONVERSATION_2),
        ConversationEntry(data=DUMMY_CONVERSATION_3),
    ]
    conversation1 = Conversation(entries=entries1)
    conversation2 = Conversation(entries=entries2)
    conversations = Conversations(conversations=[conversation1, conversation2])
    expected_list = [
        [DUMMY_CONVERSATION_1, DUMMY_CONVERSATION_2],
        [DUMMY_CONVERSATION_2, DUMMY_CONVERSATION_3],
    ]
    assert conversations.as_list() == expected_list


def test_convert_e2e_read_data_from_csv():
    converter = E2ETestConverter(path=SAMPLE_CONVERSATIONS_CSV_PATH)
    assert converter.read_file() is not None


def test_convert_e2e_data_integrity_from_csv(tmp_path: Path, data_frame: pd.DataFrame):
    file_path = tmp_path / "data_integrity.csv"
    data_frame.to_csv(file_path, index=False)

    converter = E2ETestConverter(path=str(file_path))
    assert converter.read_file() == DATA_FRAME_OUTPUT_DATA


def test_convert_e2e_read_empty_values_from_csv(
    tmp_path: Path, data_frame_empty_values: pd.DataFrame
):
    file_path = tmp_path / "data_integrity.csv"
    data_frame_empty_values.to_csv(file_path, index=False)

    converter = E2ETestConverter(path=str(file_path))
    assert converter.read_file() == DATA_FRAME_OUTPUT_DATA_EMPTY_VALUES


def test_convert_e2e_read_data_from_xlsx():
    converter = E2ETestConverter(
        path=SAMPLE_CONVERSATIONS_XLSX_PATH,
        sheet_name=SAMPLE_CONVERSATION_XLSX_SHEET_NAME,
    )
    assert converter.read_file() is not None


def test_convert_e2e_from_xlsx_without_sheet_name():
    converter = E2ETestConverter(path=SAMPLE_CONVERSATIONS_XLSX_PATH)
    with pytest.raises(RasaException, match="Please provide a sheet name"):
        converter.read_file()


def test_convert_e2e_data_integrity_from_xlsx(tmp_path: Path, data_frame: pd.DataFrame):
    file_path = tmp_path / "data_integrity.xlsx"
    data_frame.to_excel(file_path, index=False)

    converter = E2ETestConverter(
        path=str(file_path), sheet_name=SAMPLE_CONVERSATION_XLSX_SHEET_NAME
    )
    assert converter.read_file() == DATA_FRAME_OUTPUT_DATA


def test_convert_e2e_read_empty_values_from_xlsx(
    tmp_path: Path, data_frame_empty_values: pd.DataFrame
):
    file_path = tmp_path / "data_integrity.xlsx"
    data_frame_empty_values.to_excel(file_path, index=False)

    converter = E2ETestConverter(
        path=str(file_path), sheet_name=SAMPLE_CONVERSATION_XLSX_SHEET_NAME
    )
    assert converter.read_file() == DATA_FRAME_OUTPUT_DATA_EMPTY_VALUES


def test_convert_e2e_with_unsupported_extension():
    converter = E2ETestConverter(path=UNSUPPORTED_EXTENSION_PATH)
    with pytest.raises(RasaException, match="Unsupported file type"):
        converter.read_file()


def test_convert_e2e_with_directory_input_path():
    converter = E2ETestConverter(path=UNSUPPORTED_DIRECTORY_PATH)
    with pytest.raises(
        RasaException, match="The path must point to a specific file, not a directory."
    ):
        converter.read_file()


def test_convert_e2e_read_empty_csv(tmp_path: Path):
    file_path = tmp_path / "empty.csv"
    file_path.touch()

    converter = E2ETestConverter(path=str(file_path))
    with pytest.raises(
        RasaException, match="There was an error with reading the CSV file."
    ):
        converter.read_file()


def test_convert_e2e_read_empty_xlsx(tmp_path: Path):
    file_path = tmp_path / "empty.xlsx"
    file_path.touch()

    converter = E2ETestConverter(
        path=str(file_path), sheet_name=SAMPLE_CONVERSATION_XLSX_SHEET_NAME
    )
    with pytest.raises(
        RasaException, match="There was a value error while reading the file."
    ):
        converter.read_file()


def test_convert_e2e_remove_markdown_code_syntax(sample_converter: E2ETestConverter):
    markdown_string = f"```yaml\n{DUMMY_YAML_TEST}\n```"
    expected_output = DUMMY_YAML_TEST
    assert (
        sample_converter.remove_markdown_code_syntax(markdown_string) == expected_output
    )


def test_convert_e2e_split_data_into_conversations(sample_converter: E2ETestConverter):
    sample_data = [
        DUMMY_CONVERSATION_1,
        DUMMY_CONVERSATION_2,
        {},  # Empty row indicates conversation splitter
        DUMMY_CONVERSATION_3,
    ]

    expected_list = [
        [
            DUMMY_CONVERSATION_1,
            DUMMY_CONVERSATION_2,
        ],
        [
            DUMMY_CONVERSATION_3,
        ],
    ]
    assert (
        sample_converter.split_data_into_conversations(sample_data).as_list()
        == expected_list
    )


def test_convert_e2e_split_sample_conversation_data_into_conversations(
    sample_converter: E2ETestConverter,
):
    data = sample_converter.read_file()
    conversations = sample_converter.split_data_into_conversations(data)
    assert len(conversations.as_list()) == NUMBER_OF_SAMPLE_CONVERSATIONS


def test_convert_e2e_render_template(sample_converter):
    conversation_entries = [
        ConversationEntry(data=DUMMY_CONVERSATION_1),
        ConversationEntry(data=DUMMY_CONVERSATION_2),
    ]
    conversation = Conversation(entries=conversation_entries)
    sample_converter.prompt_template = "{{ conversation }}"
    rendered = sample_converter.render_template(conversation)
    expected_output = f"[{DUMMY_CONVERSATION_1}, {DUMMY_CONVERSATION_2}]"
    assert rendered == expected_output


@pytest.mark.asyncio
async def test_convert_e2e_conversations_into_tests(sample_converter):
    conversations = Conversations(
        conversations=[
            Conversation(
                entries=[
                    ConversationEntry(data=DUMMY_CONVERSATION_1),
                ]
            ),
            Conversation(
                entries=[
                    ConversationEntry(data=DUMMY_CONVERSATION_3),
                ]
            ),
        ]
    )

    with patch(
        "rasa.e2e_test.e2e_test_converter.llm_factory",
        Mock(),
    ) as mock_llm_factory:
        llm_mock = Mock()
        apredict_mock = AsyncMock(
            side_effect=[
                f"```yaml\n{DUMMY_YAML_TEST}\n```",
                f"```yaml\n{DUMMY_YAML_TEST}\n```",
            ]
        )
        llm_mock.apredict = apredict_mock
        mock_llm_factory.return_value = llm_mock
        yaml_tests_string = await sample_converter.convert_conversations_into_tests(
            conversations
        )
        expected_output = f"{DUMMY_YAML_TEST}\n{DUMMY_YAML_TEST}"
        assert yaml_tests_string.strip() == expected_output


@pytest.mark.asyncio
async def test_convert_e2e_single_conversation_into_test(sample_converter):
    conversation = Conversation(
        entries=[
            ConversationEntry(data=DUMMY_CONVERSATION_1),
        ]
    )

    with patch(
        "rasa.e2e_test.e2e_test_converter.llm_factory",
        Mock(),
    ) as mock_llm_factory:
        llm_mock = Mock()
        apredict_mock = AsyncMock(
            return_value="""```yaml\ntest_case: valid_yaml\n```"""
        )
        llm_mock.apredict = apredict_mock
        mock_llm_factory.return_value = llm_mock
        yaml_test = await sample_converter.convert_single_conversation_into_test(
            conversation
        )
        assert yaml_test.strip() == "test_case: valid_yaml"


def test_convert_e2e_yaml_is_valid(sample_converter: E2ETestConverter):
    assert sample_converter.is_yaml_valid(DUMMY_YAML_TEST) is True


def test_convert_e2e_yaml_is_not_valid(sample_converter: E2ETestConverter):
    assert sample_converter.is_yaml_valid(DUMMY_YAML_TEST[1:]) is False
