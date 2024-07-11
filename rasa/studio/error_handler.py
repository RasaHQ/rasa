import logging
from functools import wraps
from typing import Callable, Any, Dict, Type, Tuple

from keycloak.exceptions import KeycloakError
from requests.exceptions import RequestException, Timeout, ConnectionError

from rasa.shared.exceptions import RasaException
from rasa.shared.utils.cli import print_success, print_error
from rasa.studio.config import StudioConfig

logger = logging.getLogger(__name__)

# Define these constants at the module level
CALL_STEP_ERROR = (
    "Call flow reference not set. Check whether "
    "flows exist and are correctly referenced."
)
CALL_CYCLIC_ERROR = (
    "Possible CALL cyclic dependencies in flows. Please "
    "check that flows do not call each other."
)
MAX_RECURSION_ERROR = "maximum recursion depth exceeded while calling a Python object"


class ErrorHandler:
    def __init__(self) -> None:
        self.error_map: Dict[Type[Exception], Callable[[Any], Tuple[str, bool]]] = {
            RasaException: self._handle_rasa_exception,
            KeycloakError: self._handle_keycloak_error,
            ConnectionError: self._handle_connection_error,
            Timeout: self._handle_timeout_error,
            RequestException: self._handle_request_exception,
            Exception: self._handle_unexpected_error,
        }

    def handle_error(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result, success = func(*args, **kwargs)
                if self.response_has_errors(result):
                    return self._handle_graphql_errors(result)
                else:
                    print_success(result)
                    return result, success
            except Exception as e:
                for error_type, handler in self.error_map.items():
                    if isinstance(e, error_type):
                        return handler(e)
                return self._handle_unexpected_error(e)

        return wrapper

    @staticmethod
    def response_has_errors(response: Dict) -> bool:
        return (
            "errors" in response
            and isinstance(response["errors"], list)
            and len(response["errors"]) > 0
        )

    @staticmethod
    def _handle_graphql_errors(response: Dict) -> Tuple[str, bool]:
        error_msg = "Upload failed with the following errors: "
        if isinstance(response["errors"], list):
            error_details = "; ".join(
                [error.get("message", "Unknown error") for error in response["errors"]]
            )
            error_msg += f"{error_details}"
        else:
            error_msg += "No detailed error information available."

        logger.debug("Error details:")
        logger.debug(response)
        print_error(error_msg)
        return error_msg, False

    @staticmethod
    def _handle_rasa_exception(e: RasaException) -> Tuple[str, bool]:
        error_msg = f"Rasa-specific error occurred: {e!s}"
        print_error(error_msg)
        return error_msg, False

    @staticmethod
    def _handle_keycloak_error(e: KeycloakError) -> Tuple[str, bool]:
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
        return error_msg, False

    @staticmethod
    def _handle_connection_error(e: ConnectionError) -> Tuple[str, bool]:
        studio_url = StudioConfig.read_config().studio_url
        error_msg = (
            f"Unable to connect to Rasa Studio at {studio_url} \n"
            "Please check if Studio is running and the configured URL is correct. \n"
            "You may need to reconfigure Rasa Studio using 'rasa studio config'."
        )
        logger.debug("Error details:")
        logger.debug(str(e))
        print_error(error_msg)
        return error_msg, False

    @staticmethod
    def _handle_timeout_error(e: Timeout) -> Tuple[str, bool]:
        error_msg = "The request to Rasa Studio timed out. Please try again later."
        print_error(error_msg)
        return error_msg, False

    @staticmethod
    def _handle_request_exception(e: RequestException) -> Tuple[str, bool]:
        error_msg = f"An error occurred while communicating with Rasa Studio: {e!s}"
        print_error(error_msg)
        return error_msg, False

    @staticmethod
    def _handle_unexpected_error(e: Exception) -> Tuple[str, bool]:
        error_msg = f"An unexpected error occurred: {e!s}"
        print_error(error_msg)
        return error_msg, False

    @staticmethod
    def handle_calm_assistant_error(e: Exception) -> Tuple[str, bool]:
        if str(e.args[0]) == MAX_RECURSION_ERROR:
            logger.debug("Error details:")
            logger.debug(str(e))
            print_error(CALL_CYCLIC_ERROR)
            return CALL_CYCLIC_ERROR, False

        elif str(e.args[0]) == "Call flow reference not set.":
            logger.debug("Error details:")
            logger.debug(str(e))
            print_error(CALL_STEP_ERROR)
            return CALL_STEP_ERROR, False

        else:
            logger.debug("Error details:")
            logger.debug(str(e))
            print_error(f"An unexpected error occurred: {e!s}")
            return f"An unexpected error occurred: {e!s}", False

    @staticmethod
    def handle_upload_error(status: bool, response: str) -> None:
        if not status:
            print_error(f"Upload failed: {response}")

    def add_error_handler(
        self,
        error_type: Type[Exception],
        handler: Callable[[Exception], Tuple[str, bool]],
    ) -> None:
        self.error_map[error_type] = handler


error_handler: ErrorHandler = ErrorHandler()
