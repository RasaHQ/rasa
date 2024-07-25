from dataclasses import dataclass
from functools import wraps
from typing import Callable, Any, Dict

import structlog
from traceback import format_exc
from keycloak.exceptions import KeycloakError
from requests.exceptions import RequestException, Timeout, ConnectionError

from rasa.shared.exceptions import RasaException
from rasa.shared.utils.cli import print_success, print_error
from rasa.studio.config import StudioConfig

structlogger = structlog.get_logger()


@dataclass
# pick whatever name makes sense across uses, picked this one as it seems
# fitting here, but you probably can come up with a better name knowing
# the other parts of the code
class StudioResult:
    message: str
    was_successful: bool


def with_studio_error_handler(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = func(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                message, success = result
                if response_has_errors(message):
                    return _handle_graphql_errors(message)
                else:
                    print_success(message)
                    return StudioResult(message, success)
            else:
                return (
                    result  # Assuming the function might directly return a StudioResult
                )
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


def _handle_graphql_errors(response: Dict) -> StudioResult:
    if isinstance(response.get("errors"), list):
        error_details = "; ".join(
            [error.get("message", "Unknown error") for error in response["errors"]]
        )
    else:
        error_details = "No detailed error information available."

    structlogger.debug("studio.graphql_error", response=response)
    print_error(error_details)
    return StudioResult(error_details, False)


def _handle_rasa_exception(e: RasaException) -> StudioResult:
    print_error(e)
    structlogger.debug("studio.rasa_error", error=format_exc())
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
    print_error(error_msg)
    return StudioResult(error_msg, False)


def _handle_connection_error(e: ConnectionError) -> StudioResult:
    studio_url = StudioConfig.read_config().studio_url
    error_msg = (
        f"Unable to reach Rasa Studio API at {studio_url} \n"
        "Please check if Studio is running and the configured URL is correct. \n"
        "You may need to reconfigure Rasa Studio using 'rasa studio config'."
    )
    structlogger.debug("studio.graphql_error", response=str(e))
    print_error(error_msg)
    return StudioResult(error_msg, False)


def _handle_timeout_error(e: Timeout) -> StudioResult:
    error_msg = "The request timed out. Please try again later."
    print_error(error_msg)
    return StudioResult(error_msg, False)


def _handle_request_exception(e: RequestException) -> StudioResult:
    error_msg = f"An error occurred while communicating with the server: {e!s}"
    print_error(error_msg)
    return StudioResult(error_msg, False)


def _handle_unexpected_error(e: Exception) -> StudioResult:
    error_msg = f"An unexpected error occurred: {e!s}"
    print_error(error_msg)
    structlogger.debug("studio.unexpected_error", error=format_exc())
    return StudioResult(error_msg, False)
