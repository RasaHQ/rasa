from unittest.mock import patch, Mock, MagicMock

import pytest

from rasa.studio.error_handler import (
    ErrorHandler,
    RasaException,
    KeycloakError,
    ConnectionError,
    Timeout,
    RequestException,
    CALL_CYCLIC_ERROR,
    CALL_STEP_ERROR,
    MAX_RECURSION_ERROR,
)


@pytest.fixture
def error_handler():
    return ErrorHandler()


def test_handle_error_successful_execution(error_handler):
    @error_handler.handle_error
    def successful_function():
        return {"data": "success"}

    result = successful_function()
    assert result == ("Upload successful!", True)


def test_handle_error_graphql_errors(error_handler, caplog):
    @error_handler.handle_error
    def function_with_graphql_errors():
        return {
            "data": {},
            "errors": [{"message": "GraphQL error 1"}, {"message": "GraphQL error 2"}],
        }

    result = function_with_graphql_errors()
    assert result == (
        "Upload failed with the following errors: GraphQL error 1; GraphQL error 2",
        False,
    )

    assert "Upload failed with the following errors:" in caplog.text
    assert "GraphQL error 1" in caplog.text
    assert "GraphQL error 2" in caplog.text


@pytest.fixture
def mock_studio_config():
    with patch("rasa.studio.config.StudioConfig.read_config") as mock_config:
        mock_config.return_value = MagicMock(
            authentication_server_url="http://mock-auth-server:8081/auth/",
            studio_url="http://mock-studio:4000/api/graphql"
        )
        yield mock_config


@pytest.mark.parametrize(
    "exception,expected_message_template",
    [
        (RasaException("Rasa error"), "Rasa-specific error occurred: Rasa error"),
        (
            KeycloakError("Keycloak error"),
            (
                "Unable to authenticate with Keycloak at {auth_url} "
                "Error message: Keycloak error"
            ),
        ),
        (
            ConnectionError(),
            "Unable to connect to Rasa Studio at {studio_url} \n"
            "Please check if Studio is running and the configured URL is correct. \n"
            "You may need to reconfigure Rasa Studio using 'rasa studio config'.",
        ),
        (Timeout(), "The request to Rasa Studio timed out. Please try again later."),
        (
            RequestException("Request failed"),
            "An error occurred while communicating with Rasa Studio: Request failed",
        ),
        (
            Exception("Unexpected error"),
            "An unexpected error occurred: Unexpected error",
        ),
    ],
)
def test_handle_error_exceptions(error_handler, mock_studio_config, exception, expected_message_template):
    @error_handler.handle_error
    def function_with_exception():
        raise exception

    result = function_with_exception()

    config = mock_studio_config.return_value
    expected_message = expected_message_template.format(
        auth_url=config.authentication_server_url,
        studio_url=config.studio_url
    )

    assert result == (expected_message, False)


def test_response_has_errors(error_handler):
    if error_handler._response_has_errors({"errors": [{"message": "Error"}]}) is True:
        assert True
    else:
        assert False

    if error_handler._response_has_errors({"errors": []}) is False:
        assert True
    else:
        assert False

    if error_handler._response_has_errors({"data": "Success"}) is False:
        assert True
    else:
        assert False

    if error_handler._response_has_errors({"errors": None}) is False:
        assert True
    else:
        assert False


def test_add_custom_error_handler(error_handler):
    # Test case 4a: Adding a custom error handler
    # custom_exception = type("CustomException", (Exception, ""), {})
    custom_exception = Exception
    custom_handler = Mock(return_value=("Custom error handled", False))

    error_handler.add_error_handler(custom_exception, custom_handler)

    @error_handler.handle_error
    def function_with_custom_exception():
        raise custom_exception("Custom error")

    result = function_with_custom_exception()

    # Test case 4b: Verify that the custom handler is called
    custom_handler.assert_called_once()
    assert result == ("Custom error handled", False)


@patch("rasa.studio.error_handler.logger")
def test_handle_calm_assistant_error(mock_logger, error_handler):
    # Test case 5a: Handling MAX_RECURSION_ERROR
    error_handler.handle_calm_assistant_error(Exception(MAX_RECURSION_ERROR))
    mock_logger.error.assert_called_with(CALL_CYCLIC_ERROR)

    # Test case 5b: Handling "Call flow reference not set" error
    error_handler.handle_calm_assistant_error(Exception("Call flow reference not set."))
    mock_logger.error.assert_called_with(CALL_STEP_ERROR)

    # Test case 5c: Handling unexpected errors
    unexpected_error = "Unexpected error"
    error_handler.handle_calm_assistant_error(Exception(unexpected_error))
    mock_logger.error.assert_called_with(
        f"An unexpected error occurred: {unexpected_error}"
    )


@patch("rasa.studio.error_handler.logger")
def test_handle_upload_error(mock_logger, error_handler):
    error_handler.handle_upload_error(True, "Upload successful")
    mock_logger.error.assert_not_called()

    error_handler.handle_upload_error(False, "Upload failed")
    mock_logger.error.assert_called_with("Upload failed: Upload failed")
