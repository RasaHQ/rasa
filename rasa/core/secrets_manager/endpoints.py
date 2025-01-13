import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text, Union, cast

from rasa.core.secrets_manager.constants import (
    SECRET_KEY_LABEL,
    SECRET_MANAGER_PREFIX,
    SOURCE_KEY_LABEL,
    SUPPORTED_SECRET_MANAGERS,
)
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class CredentialsLocation:
    """Points to where secret is stored.

    This is a DTO object which is used to pass the configuration to the secret manager.
    More specific configuration is created based on this DTO and
    a secret manager which is used.
    For example: VaultSecretsManager will create VaultStoredSecret
    based on this DTO.
    """

    def __init__(self, source: Text, secret_key: Text, **kwargs: Optional[Any]) -> None:
        """Initialise the CredentialsLocation.

        Args:
            source: The source of the secret.
            secret_key: The key of the secret.
            **kwargs: Additional arguments.
        """
        self.source = source
        self.secret_key = secret_key
        self.kwargs = kwargs

    def get_secret_manager_name(self) -> Text:
        """Get the name of the targeted secret manager.

        Returns:
            The name of the secret manager.
        """
        return self.source.replace(f"{SECRET_MANAGER_PREFIX}.", "")

    @staticmethod
    def is_credentials_location_instance(
        value: Union[Text, "CredentialsLocation"],
    ) -> bool:
        """Check if the value is a CredentialsLocation.

        Args:
            value: The value to check.

        Returns:
            True if the value is an instance  CredentialsLocation.
        """
        return isinstance(value, CredentialsLocation)

    @staticmethod
    def is_property_non_empty_string(
        credentials_location: Dict[Text, Text], property_name: Text
    ) -> bool:
        """Check if the property of credentials location is valid.

        Args:
            credentials_location: The credentials location to check.
            property_name: The property to check.

        Returns:
            True if the property is valid.
        """
        if property_name not in credentials_location.keys():
            return False

        if not isinstance(credentials_location.get(property_name), str):
            return False

        if credentials_location.get(property_name) == "":
            return False

        return True

    @staticmethod
    def is_credentials_location_valid(
        raw_credentials_location: Dict[Text, Text],
    ) -> bool:
        """Check if the configuration is a secret manager configuration.

        Args:
            raw_credentials_location: A dictionary of values to check.

        Returns:
            True if the value is a valid credentials' location.
        """
        if not isinstance(raw_credentials_location, dict):
            return False

        if not CredentialsLocation._is_source_valid(
            raw_credentials_location=raw_credentials_location
        ):
            return False

        if not CredentialsLocation.is_property_non_empty_string(
            credentials_location=raw_credentials_location,
            property_name=SECRET_KEY_LABEL,
        ):
            return False

        return True

    @staticmethod
    def _is_source_valid(raw_credentials_location: Dict[Text, Text]) -> bool:
        """Check if the source is valid.

        Args:
            raw_credentials_location: A dictionary of values to check.

        Returns:
            True if the source is valid.
        """
        is_source_non_empty_string = CredentialsLocation.is_property_non_empty_string(
            credentials_location=raw_credentials_location,
            property_name=SOURCE_KEY_LABEL,
        )

        if not is_source_non_empty_string:
            return False

        source_value = raw_credentials_location.get(SOURCE_KEY_LABEL)
        if not source_value or source_value == "":
            return False

        parts = source_value.split(".")

        return (
            is_source_non_empty_string
            and len(parts) == 2
            and parts[0] == SECRET_MANAGER_PREFIX
            and parts[1] != ""
        )

    @staticmethod
    def are_required_properties_valid(
        value: Dict[Text, Any], endpoint_property: Text
    ) -> bool:
        """Check if required properties of raw credentials location are valid.

        Args:
            value: a raw credentials location value
            endpoint_property: name of the property to which raw
                credentials value is assigned to

        Returns:
            True if source and secret_key are valid.
            Otherwise, it returns False.
        """
        if not CredentialsLocation.is_property_non_empty_string(
            value, SOURCE_KEY_LABEL
        ):
            logger.error(
                f"Property '{SOURCE_KEY_LABEL}' is missing or invalid for "
                f"'{endpoint_property}'."
            )
            return False

        if not CredentialsLocation.is_property_non_empty_string(
            value, SECRET_KEY_LABEL
        ):
            logger.error(
                f"Property '{SECRET_KEY_LABEL}' is missing or invalid for "
                f"'{endpoint_property}'."
            )
            return False

        return True


@dataclass
class EndpointTrait:
    """Represents a trait of an endpoint.

    Args:
        endpoint_config: The config for an endpoint.
        endpoint_type: The type of the endpoint. Like tracker_store, event_broker etc.
    """

    endpoint_config: "EndpointConfig"
    endpoint_type: Text


class EndpointReader:
    """Reads and validates parts of endpoint configuration.

    This is a helper class for the `EndpointConfig` class.
    """

    def __init__(self, endpoint_config: "EndpointConfig"):
        """Initialise the endpoint reader.

        Args:
            endpoint_config: The configuration for an endpoint.
        """
        self.endpoint_config = endpoint_config

    def get_property_value(
        self,
        endpoint_property_name: Text,
    ) -> Optional[Union[str, CredentialsLocation]]:
        """Return a value stored in the endpoint's property.

        It is constrained to work only on string values and dictionaries which
        hold credentials' location.

        Args:
            endpoint_property_name: The endpoint's property name.

        Returns:
            The value of the property. It can be a string or a CredentialsLocation.
            We have constrained it to work only on string values and dictionaries
            because they are the only types which are supported by the
            endpoint's properties we are currently interested in.
        """
        endpoint_property_value = self._get_raw_property_value(endpoint_property_name)

        if (
            CredentialsLocation.is_credentials_location_valid(
                raw_credentials_location=endpoint_property_value
            )
            and endpoint_property_value.get(SOURCE_KEY_LABEL).replace(
                f"{SECRET_MANAGER_PREFIX}.", ""
            )
            in SUPPORTED_SECRET_MANAGERS
        ):
            return CredentialsLocation(
                **endpoint_property_value,
            )

        return endpoint_property_value

    def _get_raw_property_value(self, property_name: Text) -> Any:
        """Get the raw value of a property in EndpointsConfig.

        Args:
            property_name: The key of the property.

        Returns:
            The raw value of the property.
        """
        # check if it is stored in kwargs
        property_value = self.endpoint_config.kwargs.get(property_name)

        if property_value is None:
            # check if it is an attribute of EndpointsConfig
            if property_name in self.endpoint_config.__dict__.keys():
                property_value = self.endpoint_config.__dict__.get(property_name)
            else:
                raise RasaException(
                    f"Property {property_name} is not defined in the "
                    f"endpoint's configuration."
                )

        return property_value

    def has_property(
        self,
        property_name: Text,
    ) -> bool:
        """Return True if an endpoint has a given property.

        Otherwise, returns False.
        """
        return property_name in self.get_keys()

    def is_endpoint_property_valid(
        self,
        endpoint_property: Text,
    ) -> bool:
        """Validate the value of endpoint's property.

        Args:
            endpoint_property: name of the property to validate

        Returns:
            True if the value of the property is valid.
            Otherwise, it returns False.
        """
        value = self.get_property_value(endpoint_property_name=endpoint_property)
        if isinstance(value, str) and value == "":
            logger.error(f"Property '{endpoint_property}' is empty.")
            return False

        if isinstance(value, dict):
            return CredentialsLocation.are_required_properties_valid(
                value=value, endpoint_property=endpoint_property
            )

        return True

    def get_endpoint_properties_managed_by_secret_manager(
        self,
    ) -> Optional[Dict[Text, List[Text]]]:
        """Get endpoint's properties that are managed by the secret manager.

        Returns:
            A dictionary of the endpoint's properties that are managed
            by the secret manager.
            The key is the name of the secret manager and the value is a list of the
            endpoint's properties that are managed by the secret manager.

            For example, if the endpoint's properties `username` and `password`
            are managed by the Vault secret manager, the returned dictionary will be:
            {
                vault: ["username", "password"]
            }
        """
        config_keys: Dict[Text, List[Text]] = {}

        for config_key_name in self.get_keys():
            config = self.get_property_value(endpoint_property_name=config_key_name)

            if config and CredentialsLocation.is_credentials_location_instance(config):
                credentials_location = cast(CredentialsLocation, config)
                secret_manager_name = credentials_location.source.replace(
                    f"{SECRET_MANAGER_PREFIX}.", ""
                )
                if secret_manager_name not in config_keys.keys():
                    config_keys[secret_manager_name] = []

                config_keys[secret_manager_name].append(config_key_name)

        return config_keys if len(config_keys.keys()) > 0 else None

    def get_keys(self) -> List[Text]:
        """Get the list of all properties of the endpoint's configuration.

        Returns:
            The list of all properties of the endpoint's configuration.
        """
        return [key for key in self.endpoint_config.kwargs.keys()] + [
            key for key in self.endpoint_config.__dict__.keys()
        ]


class TrackerStoreEndpointValidator:
    """Processes the tracker store key's related to vault configuration."""

    USERNAME_PROPERTY = "username"
    PASSWORD_PROPERTY = "password"

    def __init__(self, endpoint_reader: EndpointReader):
        """Initialise the TrackerStoreEndpointValidator.

        Args:
            endpoint_reader: The reader of endpoint's configuration.
        """
        self.endpoint_reader = endpoint_reader

    def is_endpoint_config_valid(self) -> bool:
        """Check if the endpoint config is valid.

        Returns:
            True if the endpoint config is valid, False otherwise.
        """
        return self._is_username_valid() and self._is_password_valid()

    def _is_username_valid(self) -> bool:
        """Check if the username property is valid.

        Returns:
            True if the username property is valid, False otherwise.
        """
        return not self.endpoint_reader.has_property(
            property_name=self.USERNAME_PROPERTY
        ) or self.endpoint_reader.is_endpoint_property_valid(
            endpoint_property=self.USERNAME_PROPERTY
        )

    def _is_password_valid(self) -> bool:
        """Check if the password property is valid.

        Returns:
            True if the password property is valid, False otherwise.
        """
        return self.endpoint_reader.has_property(
            property_name=self.PASSWORD_PROPERTY
        ) and self.endpoint_reader.is_endpoint_property_valid(
            endpoint_property=self.PASSWORD_PROPERTY
        )
