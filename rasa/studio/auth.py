from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union

import jwt
from keycloak import KeycloakOpenID, KeycloakError
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.yaml import read_yaml_file, write_yaml

from rasa.studio.config import StudioConfig
from rasa.studio.constants import (
    KEYCLOAK_ACCESS_TOKEN_KEY,
    KEYCLOAK_EXPIRES_IN_KEY,
    KEYCLOAK_REFRESH_EXPIRES_IN_KEY,
    KEYCLOAK_REFRESH_TOKEN,
)
from rasa.studio.results_logger import with_studio_error_handler, StudioResult


class StudioAuth:
    def __init__(
        self,
        studio_config: StudioConfig,
    ) -> None:
        self.config = studio_config
        self.keycloak_openid = KeycloakOpenID(
            server_url=studio_config.authentication_server_url,
            client_id=studio_config.client_id,
            realm_name=studio_config.realm_name,
        )

    def health_check(self) -> bool:
        """Check if the Keycloak server is reachable.

        Returns:
        True if the server is reachable, False otherwise.
        """
        try:
            self.keycloak_openid.well_known()
            return True
        except Exception:
            return False

    @with_studio_error_handler
    def login(
        self, username: Text, password: Text, totp: Optional[int] = None
    ) -> StudioResult:
        token_dict = self.keycloak_openid.token(
            username=username, password=password, totp=totp
        )
        keycloak_token = self._resolve_token(token_dict)

        KeycloakTokenWriter.write_token_to_file(
            keycloak_token, token_file_location=DEFAULT_TOKEN_FILE_PATH
        )

        return StudioResult.success("Login successful.")

    @with_studio_error_handler
    def refresh_token(self, refresh_token: Text) -> StudioResult:
        try:
            token_dict = self.keycloak_openid.refresh_token(refresh_token)
        except Exception as e:
            raise KeycloakError(f"Could not refresh token. Error: {e}")

        keycloak_token = self._resolve_token(token_dict)

        KeycloakTokenWriter.write_token_to_file(
            keycloak_token, token_file_location=DEFAULT_TOKEN_FILE_PATH
        )

        return StudioResult.success("Token refreshed successfully.")

    @staticmethod
    def _resolve_token(token_dict: Dict[Text, Any]) -> KeycloakToken:
        return KeycloakToken(
            access_token=token_dict[KEYCLOAK_ACCESS_TOKEN_KEY],
            expires_in=token_dict[KEYCLOAK_EXPIRES_IN_KEY],
            refresh_expires_in=token_dict[KEYCLOAK_REFRESH_EXPIRES_IN_KEY],
            refresh_token=token_dict[KEYCLOAK_REFRESH_TOKEN],
            token_type=token_dict[TOKEN_TYPE_KEY],
        )


ACCESS_TOKEN_KEY = "access_token"
ACCESS_TOKEN_EXPIRATION_TIME_KEY = "access_token_expiration_time"
REFRESH_TOKEN_KEY = "refresh_token"
REFRESH_TOKEN_EXPIRATION_TIME_KEY = "refresh_token_expiration_time"
TOKEN_TYPE_KEY = "token_type"

JWT_TOKEN_EXPIRES_AT_KEY = "exp"


@dataclass
class KeycloakToken:
    access_token: Text
    expires_in: int
    refresh_expires_in: int
    refresh_token: Text
    token_type: Text

    def to_dict(self) -> Dict[Text, Union[Text, int]]:
        return {
            ACCESS_TOKEN_KEY: self.access_token,
            ACCESS_TOKEN_EXPIRATION_TIME_KEY: self.expires_in,
            REFRESH_TOKEN_KEY: self.refresh_token,
            REFRESH_TOKEN_EXPIRATION_TIME_KEY: self.refresh_expires_in,
            TOKEN_TYPE_KEY: self.token_type,
        }

    def expires_at(self) -> datetime.datetime:
        jwt_access_token = KeycloakToken._decode_token(self.access_token)
        expires_at = jwt_access_token[JWT_TOKEN_EXPIRES_AT_KEY]
        return datetime.datetime.utcfromtimestamp(expires_at)

    def is_expired(self) -> bool:
        return self.expires_at() < datetime.datetime.now()

    def can_refresh(self) -> bool:
        return self._has_refresh_token() and not self._has_refresh_token_expired()

    def _refresh_expiration_time(self) -> datetime.datetime:
        jwt_refresh_token = KeycloakToken._decode_token(self.refresh_token)
        expires_at = jwt_refresh_token[JWT_TOKEN_EXPIRES_AT_KEY]
        return datetime.datetime.utcfromtimestamp(expires_at)

    def _has_refresh_token_expired(self) -> bool:
        return self._refresh_expiration_time() < datetime.datetime.now()

    def _has_refresh_token(self) -> bool:
        return self.refresh_token is not None and self.refresh_token != ""

    @staticmethod
    def _decode_token(token: Text) -> Dict[Text, Any]:
        """Decode a JWT token."""
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except jwt.exceptions.DecodeError as e:
            raise RasaException(f"Could not decode token. Error: {e}.")


YAMLContent = Union[List[Any], Dict[Text, Any]]

DEFAULT_RASA_CONFIG_PATH = Path.home() / ".config" / "rasa"
DEFAULT_TOKEN_FILE = "studio_token.yaml"
DEFAULT_TOKEN_FILE_PATH = DEFAULT_RASA_CONFIG_PATH / DEFAULT_TOKEN_FILE


class KeycloakTokenReader:
    """Reads the token from the token file.

    If the token file does not exist, exception will be raised.
    """

    def __init__(self, token_path: Path = DEFAULT_TOKEN_FILE_PATH) -> None:
        """Initialise the token reader.

        Args:
            token_path: path to the token file.
        """
        token_file_content = self._load_token_from_cache_file(token_path)
        self.token = self._valid_token_content(token_file_content)

    def get_token(self) -> KeycloakToken:
        """Returns the token from the token file.

        Returns:
            The token from the token file.
        """
        return self.token

    @staticmethod
    def _valid_token_content(content: YAMLContent) -> KeycloakToken:
        if isinstance(content, list):
            raise ValueError(
                "Content of token file is in list format. "
                "Content should be in dictionary format. "
                "Please make sure the file is a valid YAML file."
            )

        access_token = content.get(ACCESS_TOKEN_KEY)
        if not access_token or not isinstance(access_token, str):
            raise ValueError(
                f"{ACCESS_TOKEN_KEY} in token file is not in text format. "
                "Please make sure the value is in text format."
            )

        access_token_expiration_time = content.get(ACCESS_TOKEN_EXPIRATION_TIME_KEY)
        if not isinstance(access_token_expiration_time, int):
            raise ValueError(
                f"{ACCESS_TOKEN_EXPIRATION_TIME_KEY} in "
                f"token file is not in integer format. "
                "Please make sure the value is in integer format."
            )

        refresh_token = content.get(REFRESH_TOKEN_KEY)
        if refresh_token is None:
            raise ValueError(
                f"{REFRESH_TOKEN_KEY} in token file is "
                f"not present or it is set to None. "
                "Please make sure the value is present and in text format."
            )
        elif not isinstance(refresh_token, str) or refresh_token == "":
            raise ValueError(
                f"{REFRESH_TOKEN_KEY} in token file is not in text format. "
                "Please make sure the value is in text format."
            )

        refresh_token_expiration_time = content.get(REFRESH_TOKEN_EXPIRATION_TIME_KEY)
        if refresh_token_expiration_time is None:
            raise ValueError(
                f"{REFRESH_TOKEN_EXPIRATION_TIME_KEY} in token file "
                f"is not present or it is set to None. "
                "Please make sure the value is present and in text format."
            )
        elif (
            not isinstance(refresh_token_expiration_time, int)
            or refresh_token_expiration_time == ""
        ):
            raise ValueError(
                f"{REFRESH_TOKEN_EXPIRATION_TIME_KEY} in "
                f"token file is not in text format. "
                "Please make sure the value is in text format."
            )

        token_type = content.get(TOKEN_TYPE_KEY)

        if token_type is None:
            raise ValueError(
                f"{TOKEN_TYPE_KEY} in token file is "
                f"not present or it is set to None. "
                "Please make sure the value is present and in text format."
            )
        elif not isinstance(token_type, str) or token_type == "":
            raise ValueError(
                f"{TOKEN_TYPE_KEY} in "
                f"token file is not in text format. "
                "Please make sure the value is in text format."
            )

        return KeycloakToken(
            access_token=access_token,
            expires_in=access_token_expiration_time,
            refresh_token=refresh_token,
            refresh_expires_in=refresh_token_expiration_time,
            token_type=token_type,
        )

    @staticmethod
    def _load_token_from_cache_file(
        token_path: Path,
    ) -> YAMLContent:
        if not token_path.is_file():
            raise RasaException(
                f"Could not load Keycloak token from location {token_path}."
                f"Please make sure to run `rasa studio login` first."
            )

        return read_yaml_file(token_path)


class KeycloakTokenWriter:
    @staticmethod
    def write_token_to_file(token: KeycloakToken, token_file_location: Path) -> None:
        write_yaml(token.to_dict(), token_file_location)
