import os.path
from pathlib import Path
from typing import Any, Dict, Text
from unittest.mock import MagicMock, patch

import pytest
from pykwalify.errors import SchemaError
from pytest import MonkeyPatch
from rasa.shared.utils.validation import YamlValidationException

from rasa.constants import PACKAGE_NAME
from rasa.utils.validation import read_schema_file, validate_yaml_content


@pytest.fixture
def mock_read_yaml_file(monkeypatch: MonkeyPatch) -> MagicMock:
    return MagicMock()


@pytest.fixture
def set_mock_read_yaml_file(
    monkeypatch: MonkeyPatch, mock_read_yaml_file: MagicMock
) -> None:
    monkeypatch.setattr("rasa.shared.utils.io.read_yaml_file", mock_read_yaml_file)


def test_read_schema_file(
    set_mock_read_yaml_file: None,
    mock_read_yaml_file: MagicMock,
) -> None:
    # Given
    package_path = os.path.join("canonical", "path", "to")
    input_schema_file = os.path.join("tests", "data", "test_schema.yml")
    full_path = os.path.join(package_path, input_schema_file)

    with patch("importlib_resources.files") as mock_importlib_resources_files:
        mock_importlib_resources_files.return_value = Path(package_path)

        # When
        read_schema_file(input_schema_file)

    # Then
    mock_importlib_resources_files.assert_called_with(PACKAGE_NAME)
    mock_read_yaml_file.assert_called_with(full_path)


@pytest.fixture
def mock_read_yaml(mock_read_yaml_file: MagicMock) -> MagicMock:
    return MagicMock()


@pytest.fixture
def set_mock_read_yaml(monkeypatch: MonkeyPatch, mock_read_yaml: MagicMock) -> None:
    monkeypatch.setattr("rasa.shared.utils.io.read_yaml", mock_read_yaml)


@pytest.fixture
def mock_pykwalify_core_instance() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_pykwalify_core(
    monkeypatch: MonkeyPatch, mock_pykwalify_core_instance: MagicMock
) -> MagicMock:
    core = MagicMock()
    core.return_value = mock_pykwalify_core_instance
    return core


@pytest.fixture
def pykwalify_core(monkeypatch: MonkeyPatch, mock_pykwalify_core: MagicMock) -> None:
    monkeypatch.setattr("rasa.utils.validation.Core", mock_pykwalify_core)


def test_validate_yaml_content(
    mock_pykwalify_core: MagicMock,
    mock_pykwalify_core_instance: MagicMock,
    pykwalify_core: None,
) -> None:
    test_case_file_content: Dict[Text, Any] = {}
    mock_pykwalify_core_instance.validate.return_value = None
    e2e_test_schema = ["some test schema"]

    try:
        validate_yaml_content(test_case_file_content, e2e_test_schema)
    except YamlValidationException as exc:
        assert (
            False
        ), f"'validate_yaml_content' should not have raised an exception {exc}"

    mock_pykwalify_core.assert_called_with(
        source_data=test_case_file_content, schema_data=e2e_test_schema
    )
    mock_pykwalify_core_instance.validate.assert_called_with(raise_exception=True)


def test_validate_yaml_content_with_invalid_yaml(
    set_mock_read_yaml: None,
    mock_read_yaml: MagicMock,
    mock_pykwalify_core: MagicMock,
    mock_pykwalify_core_instance: MagicMock,
    pykwalify_core: None,
) -> None:
    mock_pykwalify_core_instance.validate.side_effect = SchemaError("Invalid YAML")

    with pytest.raises(YamlValidationException):
        validate_yaml_content({}, ["some test schema"])
