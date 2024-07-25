from dataclasses import dataclass
from functools import wraps
from typing import Callable, Any, Dict

import structlog
from keycloak.exceptions import KeycloakError
from requests.exceptions import RequestException, Timeout, ConnectionError

from rasa.shared.exceptions import RasaException
from rasa.shared.utils.cli import print_success, print_error
from rasa.studio.config import StudioConfig

structlogger = structlog.get_logger()


@dataclass
class StudioResult:
    message: str
    was_successful: bool

    @staticmethod
    def success(message: str) -> "StudioResult":
        return StudioResult(message, was_successful=True)

    @staticmethod
    def error(response: Dict[str, Any]) -> "StudioResult":
        """Create a StudioResult from a GraphQL error response.

        Factory will evaluate the response and return a StudioResult with the
        appropriate message and success status.
        """
        if isinstance(response.get("errors"), list):
            error_details = "; ".join(
                [error.get("message", "Unknown error") for error in response["errors"]]
            )
        else:
            error_details = "No detailed error information available."

        structlogger.warn(
            "studio.graphql_error", event_info=error_details, response=response
        )
        return StudioResult(error_details, was_successful=False)


def with_studio_error_handler(
    func: Callable[..., StudioResult],
) -> Callable[..., StudioResult]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = func(*args, **kwargs)
            if result.was_successful:
                print_success(result.message)
            else:
                print_error(result.message)
            return result
        except RasaException as e:
            return _handle_rasa_exception(e)
        except KeycloakError as e:
            return _handle_keycloak_error(e)
        except ConnectionError as e:
            return _handle_connection_error(e)
        except Timeout as e:
            return _handle_timeout_error(e)
        except RequestException as e:
            return _handle_request_exception(e)
        except Exception as e:
            return _handle_unexpected_error(e)

    return wrapper


def response_has_errors(response: Dict) -> bool:
    return (
        "errors" in response
        and isinstance(response["errors"], list)
        and len(response["errors"]) > 0
    )


def _handle_rasa_exception(e: RasaException) -> StudioResult:
    error_msg = "Rasa internal exception was raised while interacting with Studio."
    structlogger.error("studio.rasa_error", event_info=error_msg, exception=str(e))
    return StudioResult(message=str(e), was_successful=False)


def _handle_keycloak_error(e: KeycloakError) -> StudioResult:
    error_msg = (
        f"Unable to authenticate with Keycloak at "
        f"{StudioConfig.read_config().authentication_server_url} "
    )
    if e.response_code == 401:
        error_msg += "Please check if the credentials are correct."
    elif e.error_message.startswith("Can't connect to server"):
        error_msg += (
            "Please check if the server is running "
            "and the configured URL is correct. \n"
            "You may need to reconfigure Rasa Studio "
            "using 'rasa studio config'."
        )
    else:
        error_msg += f"Error message: {e.error_message}"
    structlogger.error("studio.keycloak_error", event_info=error_msg, exception=str(e))
    return StudioResult(error_msg, was_successful=False)


def _handle_connection_error(e: ConnectionError) -> StudioResult:
    studio_url = StudioConfig.read_config().studio_url
    error_msg = (
        f"Unable to reach Rasa Studio API at {studio_url} \n"
        "Please check if Studio is running and the configured URL is correct. \n"
        "You may need to reconfigure Rasa Studio using 'rasa studio config'."
    )
    structlogger.error("studio.graphql_error", event_info=error_msg, exception=str(e))
    return StudioResult(error_msg, was_successful=False)


def _handle_timeout_error(e: Timeout) -> StudioResult:
    error_msg = "The request timed out. Please try again later."
    structlogger.error("studio.graphql_timeout", event_info=error_msg, exception=str(e))
    return StudioResult(error_msg, was_successful=False)


def _handle_request_exception(e: RequestException) -> StudioResult:
    error_msg = f"An error occurred while communicating with the server: {e!s}"
    structlogger.error(
        "studio.graphql.request_exception", event_info=error_msg, exception=str(e)
    )
    return StudioResult(error_msg, was_successful=False)


def _handle_unexpected_error(e: Exception) -> StudioResult:
    error_msg = f"An unexpected error occurred: {e!s}"
    structlogger.exception(
        "studio.unexpected_error", event_info=error_msg, exception=str(e)
    )
    return StudioResult(error_msg, was_successful=False)
