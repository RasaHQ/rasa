from typing import Any, Dict, Text
from unittest.mock import MagicMock

import pytest
from rasa.shared.core.domain import Domain

from rasa.markers.marker import OrMarker
from rasa.markers.marker_base import Marker
from rasa.markers.validate import validate_markers


@pytest.mark.parametrize(
    "domain_yaml, marker_yaml",
    [
        (
            """
            version: "3.1"
            intents:
              - mood_great
              - mood_unhappy
            """,
            {
                "marker_mood_expressed": {
                    "description": "Mood expressed was either unhappy or great",
                    "or": [{"intent": "mood_unhappy"}, {"intent": "mood_great"}],
                }
            },
        ),
    ],
)
def test_validate_marker(domain_yaml: Text, marker_yaml: Dict[Text, Any]) -> None:
    domain = Domain.from_yaml(domain_yaml)
    marker_name = "marker_mood_expressed"
    marker = Marker.from_config(marker_yaml.get(marker_name), name=marker_name)
    or_marker = OrMarker(markers=[marker])
    validate_markers(domain, or_marker)


@pytest.fixture
def mock_print_error_and_exit(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    _mock_print_error_and_exit = MagicMock()
    monkeypatch.setattr(
        "rasa.markers.validate.print_error_and_exit", _mock_print_error_and_exit
    )
    return _mock_print_error_and_exit


@pytest.mark.parametrize(
    "domain_yaml, marker_yaml",
    [
        (
            """
            version: "3.1"
            intents:
              - mood_great
            """,
            {
                "marker_mood_expressed": {
                    "description": "Mood expressed was either unhappy or great",
                    "or": [{"intent": "mood_unhappy"}, {"intent": "mood_great"}],
                }
            },
        ),
    ],
)
def test_validate_marker_fails(
    domain_yaml: Text,
    marker_yaml: Dict[Text, Any],
    mock_print_error_and_exit: MagicMock,
) -> None:
    domain = Domain.from_yaml(domain_yaml)
    marker_name = "marker_mood_expressed"
    marker = Marker.from_config(marker_yaml.get(marker_name), name=marker_name)
    or_marker = OrMarker(markers=[marker])
    validate_markers(domain, or_marker)
    mock_print_error_and_exit.assert_called_once_with(
        "Validation errors were found in the markers definition. "
        "Please see errors listed above and fix before running again."
    )
