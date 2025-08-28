from typing import Any, Dict, List, Optional

import pytest

from rasa.builder.exceptions import ValidationError


class TestValidationError:
    """Test cases for ValidationError class."""

    @pytest.fixture
    def sample_validation_logs(self) -> List[Dict[str, Any]]:
        """Sample validation logs for testing."""
        return [
            {"log_level": "error", "message": "Error 1", "file": "domain.yml"},
            {"log_level": "error", "message": "Error 2", "file": "stories.yml"},
            {"log_level": "warning", "message": "Warning 1", "file": "domain.yml"},
            {"log_level": "info", "message": "Info 1", "file": "config.yml"},
            {"log_level": "debug", "message": "Debug 1", "file": "nlu.yml"},
        ]

    @pytest.mark.parametrize(
        "log_levels,expected_count",
        [
            (["error"], 2),
            (["warning"], 1),
            (["info"], 1),
            (["debug"], 1),
            (["critical"], 0),
            (["trace"], 0),
            (["error", "warning"], 3),
            (["info", "debug"], 2),
            (["error", "warning", "info"], 4),
        ],
    )
    def test_get_logs(
        self,
        sample_validation_logs: List[Dict[str, Any]],
        log_levels: List[str],
        expected_count: int,
    ):
        """Test get_logs method filters logs by log levels correctly."""
        validation_error = ValidationError("Test error", sample_validation_logs)
        logs = validation_error.get_logs(log_levels)

        assert len(logs) == expected_count
        for log in logs:
            assert log["log_level"] in log_levels

    def test_get_logs_empty_validation_logs(self):
        """Test get_logs returns empty list when no validation logs exist."""
        validation_error = ValidationError("Test error")
        logs = validation_error.get_logs(["error"])

        assert logs == []

    @pytest.mark.parametrize(
        "log_levels,expected_logs_present",
        [
            (
                None,
                ["Error 1", "Error 2", "Warning 1", "Info 1", "Debug 1"],
            ),  # All logs
            (["error"], ["Error 1", "Error 2"]),
            (["warning"], ["Warning 1"]),
            (["error", "warning"], ["Error 1", "Error 2", "Warning 1"]),
            (["info", "debug"], ["Info 1", "Debug 1"]),
            ([], []),  # No logs when empty list
        ],
    )
    def test_get_error_message_with_logs(
        self,
        sample_validation_logs: List[Dict[str, Any]],
        log_levels: Optional[List[str]],
        expected_logs_present: List[str],
    ):
        """Test get_error_message_with_logs method includes correct logs based on log levels."""  # noqa: E501
        validation_error = ValidationError("Test error message", sample_validation_logs)
        error_message = validation_error.get_error_message_with_logs(log_levels)

        # Check that the base error message is included
        assert "Test error message" in error_message

        # Check that "Validation Logs" section is present when there are logs
        if expected_logs_present:
            assert "Validation Logs:" in error_message
        else:
            assert "Validation Logs:" not in error_message

        # Check that expected logs are present
        for log in expected_logs_present:
            assert log in error_message

    def test_get_error_message_with_logs_empty_validation_logs(self):
        """Test get_error_message_with_logs when no validation logs exist."""
        validation_error = ValidationError("Test error")
        error_message = validation_error.get_error_message_with_logs()

        assert error_message == "Test error"
        assert "Validation Logs:" not in error_message

    def test_get_error_message_with_logs_empty_log_levels_list(
        self, sample_validation_logs: List[Dict[str, Any]]
    ):
        """Test get_error_message_with_logs with empty log_levels list."""
        validation_error = ValidationError("Test error", sample_validation_logs)
        error_message = validation_error.get_error_message_with_logs([])

        assert "Test error" in error_message
        assert "Validation Logs:" not in error_message

    def test_get_error_message_with_logs_filtering_works(
        self, sample_validation_logs: List[Dict[str, Any]]
    ):
        """Test that get_error_message_with_logs correctly filters logs."""
        validation_error = ValidationError("Test error", sample_validation_logs)

        # Test with only error logs
        error_message = validation_error.get_error_message_with_logs(["error"])

        assert "Test error" in error_message
        assert "Error 1" in error_message
        assert "Error 2" in error_message
        assert "Warning 1" not in error_message
        assert "Info 1" not in error_message
        assert "Debug 1" not in error_message

        # Test with only warning logs
        error_message = validation_error.get_error_message_with_logs(["warning"])

        assert "Test error" in error_message
        assert "Warning 1" in error_message
        assert "Error 1" not in error_message
        assert "Error 2" not in error_message
        assert "Info 1" not in error_message
        assert "Debug 1" not in error_message
