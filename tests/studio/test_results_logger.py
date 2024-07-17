from unittest.mock import patch, Mock, MagicMock

import pytest

from rasa.studio.results_logger import (
    ResultsLogger,
    RasaException,
    KeycloakError,
    ConnectionError,
    Timeout,
    RequestException,
)


@pytest.fixture
def results_logger():
    return ResultsLogger()


def test_handle_error_successful_execution(results_logger):
    @results_logger.wrap
    def successful_function():
        return "Upload successful!", True

    result = successful_function()
    assert result == ("Upload successful!", True)


def test_handle_error_graphql_errors(results_logger, caplog):
    @results_logger.wrap
    def function_with_graphql_errors():
        return {
            "data": {},
            "errors": [{"message": "GraphQL error 1"}, {"message": "GraphQL error 2"}],
        }, False

    result = function_with_graphql_errors()
    assert result == (
        "Upload failed with the following errors: GraphQL error 1; GraphQL error 2",
        False,
    )


@pytest.fixture
def mock_studio_config():
    with patch("rasa.studio.config.StudioConfig.read_config") as mock_config:
        mock_config.return_value = MagicMock(
            authentication_server_url="http://mock-auth-server:8081/auth/",
            studio_url="http://mock-studio:4000/api/graphql",
        )
        yield mock_config


@pytest.mark.parametrize(
    "exception,expected_message_template",
    [
        (RasaException("Rasa error"), "Error: Rasa error"),
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
        (Timeout(), "The request timed out. Please try again later."),
        (
            RequestException("Request failed"),
            "An error occurred while communicating with the server: Request failed",
        ),
        (
            Exception("Unexpected error"),
            "An unexpected error occurred: Unexpected error",
        ),
    ],
)
def test_handle_error_exceptions(
    results_logger, mock_studio_config, exception, expected_message_template
):
    @results_logger.wrap
    def function_with_exception():
        raise exception

    result = function_with_exception()

    config = mock_studio_config.return_value
    expected_message = expected_message_template.format(
        auth_url=config.authentication_server_url, studio_url=config.studio_url
    )

    assert result == (expected_message, False)


def test_response_has_errors(results_logger):
    if results_logger.response_has_errors({"errors": [{"message": "Error"}]}) is True:
        assert True
    else:
        assert False

    if results_logger.response_has_errors({"errors": []}) is False:
        assert True
    else:
        assert False

    if results_logger.response_has_errors({"data": "Success"}) is False:
        assert True
    else:
        assert False

    if results_logger.response_has_errors({"errors": None}) is False:
        assert True
    else:
        assert False


def test_add_custom_results_logger(results_logger):
    custom_exception = Exception
    custom_handler = Mock(return_value=("Custom error handled", False))

    results_logger.add_error_handler(custom_exception, custom_handler)

    @results_logger.wrap
    def function_with_custom_exception():
        raise custom_exception("Custom error")

    result = function_with_custom_exception()

    custom_handler.assert_called_once()
    assert result == ("Custom error handled", False)
