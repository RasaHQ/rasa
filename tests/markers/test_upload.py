import copy
from pathlib import Path
from typing import Any, Dict, Text
from unittest.mock import MagicMock

import pytest
import requests
from _pytest.monkeypatch import MonkeyPatch
from rasa.shared.core.domain import Domain
from requests import Response

from rasa.markers.upload import (
    PATTERNS_PATH,
    _convert_marker_config_to_json,
    _convert_yaml_to_json,
    upload,
)


@pytest.fixture
def mock_validate_marker_file(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_validate_marker_file = MagicMock()
    monkeypatch.setattr(
        "rasa.markers.upload.validate_marker_file", mock_validate_marker_file
    )
    return mock_validate_marker_file


@pytest.fixture
def mock_collect_configs_from_yaml_files(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_collect_configs_from_yaml_files = MagicMock()
    monkeypatch.setattr(
        "rasa.markers.upload.collect_configs_from_yaml_files",
        mock_collect_configs_from_yaml_files,
    )
    return mock_collect_configs_from_yaml_files


@pytest.fixture
def mock_request(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_request = MagicMock()
    mock_request.post = MagicMock()
    monkeypatch.setattr("rasa.markers.upload.requests", mock_request)
    return mock_request


@pytest.fixture
def mock_collect_yaml_files_from_path(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_collect_yaml_files_from_path = MagicMock()
    mock_collect_yaml_files_from_path.return_value = ["test.yml"]

    monkeypatch.setattr(
        "rasa.markers.upload.collect_yaml_files_from_path",
        mock_collect_yaml_files_from_path,
    )
    return mock_collect_yaml_files_from_path


@pytest.mark.parametrize(
    "input_url, input_domain, input_markers_path, marker_dict, response",
    [
        (
            "http://example.com",
            MagicMock(),
            MagicMock(spec=Path),
            {
                "file.yaml": {
                    "magic_marker": {
                        "description": "This is a magic marker",
                        "or": [
                            {"intent": "greet"},
                            {"intent": "goodbye"},
                        ],
                    }
                }
            },
            {
                "status_code": 200,
                "text": """{"patterns": {
                    "count": 1,
                    "updated": 0,
                    "inserted": 1,
                    "deleted": 0
                }}""",
            },
        ),
    ],
)
def test_upload_markers(
    input_url: Text,
    input_domain: Domain,
    input_markers_path: Path,
    marker_dict: Dict[Text, Any],
    response: Dict[Text, Any],
    mock_validate_marker_file: MagicMock,
    mock_collect_yaml_files_from_path: MagicMock,
    mock_collect_configs_from_yaml_files: MagicMock,
    mock_request: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    # Set up mock objects
    post_response = Response()
    post_response.status_code = response["status_code"]
    post_response._content = response["text"].encode("utf-8")

    mock_request.post.return_value = post_response
    mock_collect_configs_from_yaml_files.return_value = marker_dict
    marker_dict_copy = copy.deepcopy(marker_dict)

    # Call the function being tested
    upload(input_url, input_domain, input_markers_path)

    # Verify that helper functions were called with correct parameters
    mock_validate_marker_file.assert_called_once_with(input_domain, input_markers_path)
    mock_collect_yaml_files_from_path.assert_called_once_with(input_markers_path)
    mock_collect_configs_from_yaml_files.assert_called_once_with(["test.yml"])
    mock_request.post.assert_called_once_with(
        f"{input_url}{PATTERNS_PATH}",
        json={"patterns": _convert_yaml_to_json(marker_dict_copy)},
    )


@pytest.fixture
def mock_print_error_and_exit(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_print_error_and_exit = MagicMock()
    monkeypatch.setattr(
        "rasa.markers.upload.print_error_and_exit", mock_print_error_and_exit
    )
    return mock_print_error_and_exit


@pytest.mark.parametrize(
    "input_url, input_domain, input_markers_path, marker_dict, response",
    [
        (
            "http://example.com",
            MagicMock(),
            MagicMock(spec=Path),
            {
                "file.yaml": {
                    "magic_marker": {
                        "description": "This is a magic marker",
                        "or": [
                            {"intent": "greet"},
                            {"intent": "goodbye"},
                        ],
                    }
                }
            },
            {
                "status_code": 400,
                "text": "Marker could not be uploaded",
            },
        ),
    ],
)
def test_upload_with_error(
    input_url: Text,
    input_domain: Domain,
    input_markers_path: Path,
    marker_dict: Dict[Text, Any],
    response: Dict[Text, Any],
    mock_validate_marker_file: MagicMock,
    mock_collect_yaml_files_from_path: MagicMock,
    mock_collect_configs_from_yaml_files: MagicMock,
    mock_request: MagicMock,
    mock_print_error_and_exit: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    # Set up mock objects
    post_response = Response()
    post_response.status_code = response["status_code"]
    post_response._content = response["text"].encode("utf-8")

    mock_request.post.return_value = post_response
    mock_collect_configs_from_yaml_files.return_value = marker_dict
    marker_dict_copy = copy.deepcopy(marker_dict)

    # Call the function being tested
    upload(input_url, input_domain, input_markers_path)

    # Verify that helper functions were called with correct parameters
    mock_validate_marker_file.assert_called_once_with(input_domain, input_markers_path)
    mock_collect_yaml_files_from_path.assert_called_once_with(input_markers_path)
    mock_collect_configs_from_yaml_files.assert_called_once_with(["test.yml"])

    upload_url = f"{input_url}{PATTERNS_PATH}"
    mock_request.post.assert_called_once_with(
        upload_url,
        json={"patterns": _convert_yaml_to_json(marker_dict_copy)},
    )

    mock_print_error_and_exit.assert_called_once_with(
        f"Failed to upload markers to {upload_url}. "
        f"Status Code: {response.get('status_code')} "
        f"Response: {response.get('text')}"
    )


@pytest.mark.parametrize(
    "marker_name, marker_config, expected",
    [
        (
            "magic_marker",
            {
                "description": "This is a magic marker",
                "or": [
                    {"intent": "greet"},
                    {"intent": "goodbye"},
                ],
            },
            {
                "name": "magic_marker",
                "description": "This is a magic marker",
                "config": {
                    "or": [
                        {"intent": "greet"},
                        {"intent": "goodbye"},
                    ],
                },
            },
        ),
        (
            "magic_marker",
            {
                "or": [
                    {"intent": "greet"},
                    {"intent": "goodbye"},
                ],
            },
            {
                "name": "magic_marker",
                "description": None,
                "config": {
                    "or": [
                        {"intent": "greet"},
                        {"intent": "goodbye"},
                    ],
                },
            },
        ),
    ],
)
def test_convert_marker_config_to_json(
    marker_name: Text, marker_config: Dict[Text, Any], expected: Dict[Text, Any]
) -> None:
    assert _convert_marker_config_to_json(marker_name, marker_config) == expected


@pytest.mark.parametrize(
    "input_url, input_domain, input_markers_path, marker_dict",
    [
        (
            "http://example.com",
            MagicMock(),
            MagicMock(spec=Path),
            {
                "file.yaml": {
                    "magic_marker": {
                        "description": "This is a magic marker",
                        "or": [
                            {"intent": "greet"},
                            {"intent": "goodbye"},
                        ],
                    }
                }
            },
        ),
    ],
)
def test_upload_with_connection_error(
    input_url: Text,
    input_domain: Domain,
    input_markers_path: Path,
    marker_dict: Dict[Text, Any],
    mock_validate_marker_file: MagicMock,
    mock_collect_yaml_files_from_path: MagicMock,
    mock_collect_configs_from_yaml_files: MagicMock,
    mock_print_error_and_exit: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    # Set up mock objects
    # Mock the requests.post function to raise a ConnectionError
    def mock_post(url: str, json: dict) -> None:
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(requests, "post", mock_post)
    mock_collect_configs_from_yaml_files.return_value = marker_dict

    # Call the function being tested
    upload(input_url, input_domain, input_markers_path)

    # Assert that the correct error message is printed
    upload_url = f"{input_url}{PATTERNS_PATH}"
    mock_print_error_and_exit.assert_called_once_with(
        f"Failed to connect to Rasa Pro Services at {upload_url}. "
        f"Make sure the server is running and the correct URL is configured."
    )
