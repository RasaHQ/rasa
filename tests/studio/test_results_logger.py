from unittest.mock import patch, MagicMock

import pytest
from keycloak import KeycloakError
from requests import RequestException, Timeout, ConnectionError

from rasa.shared.exceptions import RasaException
from rasa.studio.results_logger import with_studio_error_handler, StudioResult


def test_handle_error_successful_execution():
    @with_studio_error_handler
    def successful_function():
        return StudioResult(message="Upload successful!", was_successful=True)

    result = successful_function()
    assert result == StudioResult("Upload successful!", True)


def test_handle_error_graphql_errors():
    @with_studio_error_handler
    def function_with_graphql_errors():
        return {
            "data": {},
            "errors": [{"message": "GraphQL error 1"}, {"message": "GraphQL error 2"}],
        }, False

    result = function_with_graphql_errors()
    assert result == StudioResult(
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
    "exception,expected_result",
    [
        (RasaException("Rasa error"), StudioResult("Error: Rasa error", False)),
        (
            KeycloakError("Keycloak error"),
            StudioResult(
                "Unable to authenticate with Keycloak at "
                "http://mock-auth-server:8081/auth/ "
                "Error message: Keycloak error",
                False,
            ),
        ),
        (
            ConnectionError(),
            StudioResult(
                "Unable to connect to Rasa Studio at "
                "http://mock-studio:4000/api/graphql \n"
                "Please check if Studio is running and "
                "the configured URL is correct. \n"
                "You may need to reconfigure Rasa Studio "
                "using 'rasa studio config'.",
                False,
            ),
        ),
        (
            Timeout(),
            StudioResult("The request timed out. Please try again later.", False),
        ),
        (
            RequestException("Request failed"),
            StudioResult(
                "An error occurred while communicating with the server: Request failed",
                False,
            ),
        ),
        (
            Exception("Unexpected error"),
            StudioResult("An unexpected error occurred: Unexpected error", False),
        ),
    ],
)
def test_handle_error_exceptions(mock_studio_config, exception, expected_result):
    @with_studio_error_handler
    def function_with_exception():
        raise exception

    result = function_with_exception()
    assert result == expected_result


def test_response_has_errors():
    from rasa.studio.results_logger import response_has_errors

    assert response_has_errors({"errors": [{"message": "Error"}]})
    assert not response_has_errors({"errors": []})
    assert not response_has_errors({"data": "Success"})
    assert not response_has_errors({"errors": None})
