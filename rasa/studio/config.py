from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Text

from rasa.studio.constants import (
    RASA_STUDIO_AUTH_SERVER_URL_ENV,
    RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV,
    RASA_STUDIO_CLI_DISABLE_VERIFY_KEY_ENV,
    RASA_STUDIO_CLI_REALM_NAME_KEY_ENV,
    RASA_STUDIO_CLI_STUDIO_URL_ENV,
    STUDIO_CONFIG_KEY,
)
from rasa.utils.common import read_global_config_value, write_global_config_value

AUTH_SERVER_URL_KEY = "authentication_server_url"
STUDIO_URL_KEY = "studio_url"
CLIENT_ID_KEY = "client_id"
REALM_NAME_KEY = "realm_name"
CLIENT_SECRET_KEY = "client_secret"
DISABLE_VERIFY = "disable_verify"


@dataclass
class StudioConfig:
    authentication_server_url: Optional[Text]
    studio_url: Optional[Text]
    client_id: Optional[Text]
    realm_name: Optional[Text]
    disable_verify: bool = False

    def to_dict(self) -> Dict[Text, Optional[Any]]:
        return {
            AUTH_SERVER_URL_KEY: self.authentication_server_url,
            STUDIO_URL_KEY: self.studio_url,
            CLIENT_ID_KEY: self.client_id,
            REALM_NAME_KEY: self.realm_name,
            DISABLE_VERIFY: self.disable_verify,
        }

    @classmethod
    def from_dict(cls, data: Dict[Text, Text]) -> StudioConfig:
        return cls(
            authentication_server_url=data[AUTH_SERVER_URL_KEY],
            studio_url=data[STUDIO_URL_KEY],
            client_id=data[CLIENT_ID_KEY],
            realm_name=data[REALM_NAME_KEY],
            disable_verify=data.get(DISABLE_VERIFY, False),
        )

    def write_config(self) -> None:
        write_global_config_value(
            STUDIO_CONFIG_KEY,
            self.to_dict(),
        )

    @staticmethod
    def read_config() -> StudioConfig:
        env_config = StudioConfig._read_env_config()
        file_config = StudioConfig._read_config_from_file()

        return env_config._merge(file_config)

    def is_valid(self) -> bool:
        return all(
            [
                self.authentication_server_url,
                self.studio_url,
                self.client_id,
                self.realm_name,
            ]
        )

    @staticmethod
    def _read_config_from_file() -> StudioConfig:
        config = read_global_config_value(STUDIO_CONFIG_KEY, unavailable_ok=True)

        if config is None:
            return StudioConfig(None, None, None, None, False)

        if not isinstance(config, dict):
            raise ValueError(
                "Invalid config file format. "
                "Expected a dictionary, but found a {}."
                "".format(type(config).__name__)
            )

        for key in config:
            if not isinstance(config[key], str) and key != DISABLE_VERIFY:
                raise ValueError(
                    "Invalid config file format. "
                    f"Key '{key}' is not a text value."
                    "Please provide a valid value for the key."
                    ""
                )

        return StudioConfig.from_dict(config)

    @staticmethod
    def _read_env_config() -> StudioConfig:
        return StudioConfig(
            authentication_server_url=StudioConfig._read_env_value(
                RASA_STUDIO_AUTH_SERVER_URL_ENV
            ),
            studio_url=StudioConfig._read_env_value(RASA_STUDIO_CLI_STUDIO_URL_ENV),
            client_id=StudioConfig._read_env_value(RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV),
            realm_name=StudioConfig._read_env_value(RASA_STUDIO_CLI_REALM_NAME_KEY_ENV),
            disable_verify=bool(
                os.getenv(RASA_STUDIO_CLI_DISABLE_VERIFY_KEY_ENV, False)
            ),
        )

    @staticmethod
    def _read_env_value(env_name: Text) -> Optional[Text]:
        value = os.getenv(env_name, None)
        if value == "":
            raise ValueError(
                "Invalid config file format. "
                f"Key '{env_name}' is not a text value."
                "Please provide a valid value for the key."
                ""
            )
        return value

    def _merge(self, other: StudioConfig) -> StudioConfig:
        return StudioConfig(
            authentication_server_url=(
                self.authentication_server_url or other.authentication_server_url
            ),
            studio_url=self.studio_url or other.studio_url,
            client_id=self.client_id or other.client_id,
            realm_name=self.realm_name or other.realm_name,
            disable_verify=self.disable_verify or other.disable_verify,
        )
