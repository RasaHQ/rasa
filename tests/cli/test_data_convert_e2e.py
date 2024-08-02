from pathlib import Path

import pandas as pd
import pytest

from rasa.e2e_test.data_convert_e2e import E2ETestConverter
from rasa.shared.exceptions import RasaException

SAMPLE_CONVERSATIONS_CSV_PATH = "data/test_data_convert_e2e/sample_conversations.csv"
SAMPLE_CONVERSATIONS_XLSX_PATH = "data/test_data_convert_e2e/sample_conversations.xlsx"
SAMPLE_CONVERSATION_XLSX_SHEET_NAME = "Sheet1"
UNSUPPORTED_EXTENSION_PATH = "sample_conversations.txt"
UNSUPPORTED_DIRECTORY_PATH = "/data/sample_conversations/"

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


@pytest.fixture(autouse=True)
def data_frame() -> pd.DataFrame:
    return pd.DataFrame(DATA_FRAME_INPUT_DATA)


@pytest.fixture(autouse=True)
def data_frame_empty_values() -> pd.DataFrame:
    return pd.DataFrame(DATA_FRAME_INPUT_DATA_EMPTY_VALUES)


def test_convert_e2e_read_data_from_csv():
    converter = E2ETestConverter(path=SAMPLE_CONVERSATIONS_CSV_PATH)
    converter.read_file()
    assert converter.data is not None


def test_convert_e2e_data_integrity_from_csv(tmp_path: Path, data_frame: pd.DataFrame):
    file_path = tmp_path / "data_integrity.csv"
    data_frame.to_csv(file_path, index=False)

    converter = E2ETestConverter(path=str(file_path))
    converter.read_file()
    assert converter.data == DATA_FRAME_OUTPUT_DATA


def test_convert_e2e_read_empty_values_from_csv(
    tmp_path: Path, data_frame_empty_values: pd.DataFrame
):
    file_path = tmp_path / "data_integrity.csv"
    data_frame_empty_values.to_csv(file_path, index=False)

    converter = E2ETestConverter(path=str(file_path))
    converter.read_file()
    assert converter.data == DATA_FRAME_OUTPUT_DATA_EMPTY_VALUES


def test_convert_e2e_read_data_from_xlsx():
    converter = E2ETestConverter(
        path=SAMPLE_CONVERSATIONS_XLSX_PATH,
        sheet_name=SAMPLE_CONVERSATION_XLSX_SHEET_NAME,
    )
    converter.read_file()
    assert converter.data is not None


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
    converter.read_file()
    assert converter.data == DATA_FRAME_OUTPUT_DATA


def test_convert_e2e_read_empty_values_from_xlsx(
    tmp_path: Path, data_frame_empty_values: pd.DataFrame
):
    file_path = tmp_path / "data_integrity.xlsx"
    data_frame_empty_values.to_excel(file_path, index=False)

    converter = E2ETestConverter(
        path=str(file_path), sheet_name=SAMPLE_CONVERSATION_XLSX_SHEET_NAME
    )
    converter.read_file()
    assert converter.data == DATA_FRAME_OUTPUT_DATA_EMPTY_VALUES


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
    with pytest.raises(RasaException, match="The file could not be read."):
        converter.read_file()


def test_convert_e2e_read_empty_xlsx(tmp_path: Path):
    file_path = tmp_path / "empty.xlsx"
    file_path.touch()

    converter = E2ETestConverter(
        path=str(file_path), sheet_name=SAMPLE_CONVERSATION_XLSX_SHEET_NAME
    )
    with pytest.raises(RasaException, match="The file could not be read."):
        converter.read_file()
