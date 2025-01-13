from __future__ import annotations

import abc
import copy
import logging
from typing import Any, ClassVar, Dict, List, Optional, Text, cast

from rasa.core.secrets_manager.endpoints import (
    CredentialsLocation,
    EndpointReader,
    EndpointTrait,
)
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig
from rasa.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class SecretManagerConfig:
    """Configuration for the secret manager.

    This is used as DTO to pass the configuration to the secret manager.
    """

    def __init__(self, secret_manager_type: Text):
        """Initialise the secret manager configuration.

        Args:
            secret_manager_type: The type of the secret manager.
        """
        self.secret_manager_type = secret_manager_type


class SecretsManager:
    """Represents a secrets manager."""

    @abc.abstractmethod
    def load_secrets(self, endpoint_config: EndpointTrait) -> Dict[Text, Any]:
        """Load secrets from the secret manager."""
        ...

    @abc.abstractmethod
    def name(self) -> Text:
        """Return the name of the secret manager."""
        ...


class SecretsManagerProvider(metaclass=Singleton):
    """Represents a provider for secrets managers."""

    secret_managers: ClassVar[Dict[Text, SecretsManager]] = {}

    def register_secret_manager(self, manager: Optional[SecretsManager]) -> None:
        """Register a secret manager.

        Args:
            manager: The secrets manager to register.
        """
        if manager is None:
            return

        self.secret_managers[manager.name()] = manager

    def get_secrets_manager(self, manager_name: Text) -> Optional[SecretsManager]:
        """Get a secrets manager by name.

        Args:
            manager_name: The name of the secret manager.

        Returns:
            The secret manager.
        """
        manager = self.secret_managers.get(manager_name)

        return manager


class EndpointResolver:
    """Resolves endpoints configuration with values from secret managers."""

    @staticmethod
    def _get_secret_manager(name: Text) -> Optional[SecretsManager]:
        secret_manager = SecretsManagerProvider().get_secrets_manager(manager_name=name)

        if secret_manager is None:
            raise RasaException(
                f"Failed to load secret manager '{name}'. "
                f"Please check the secret manager configuration "
                f"in `endpoints.yml` file and environment variables."
            )

        return secret_manager

    @staticmethod
    def _get_credentials_from_secret_managers(
        endpoint_config: EndpointConfig,
    ) -> Optional[Dict[Text, Text]]:
        """Get credentials from secret managers.

        It first groups endpoint properties by secret manager.
        Then it gets credentials from each secret manager.

        Args:
            endpoint_config: Endpoint configuration.

        Returns:
            Credentials from secret manager.
                Example: {"username": "some user", "password": "some password"}
        """
        endpoint_properties_per_secret_manager = EndpointReader(
            endpoint_config=endpoint_config
        ).get_endpoint_properties_managed_by_secret_manager()

        if not endpoint_properties_per_secret_manager:
            return None

        credentials: Dict[Text, Text] = {}
        for (
            secret_manager_source,
            config_keys,
        ) in endpoint_properties_per_secret_manager.items():
            credentials_from_secret_manager = (
                EndpointResolver._get_credentials_from_secret_manager(
                    endpoint_properties=config_keys,
                    endpoint_config=endpoint_config,
                    secret_manager_name=secret_manager_source,
                )
            )

            if credentials_from_secret_manager:
                credentials.update(credentials_from_secret_manager)

        return credentials if credentials else None

    @staticmethod
    def _get_credentials_from_secret_manager(
        endpoint_properties: List[Text],
        endpoint_config: "EndpointConfig",
        secret_manager_name: Text,
    ) -> Optional[Dict[Text, Text]]:
        """Get credentials from one secret manager.

        Args:
            endpoint_properties: Endpoint's properties.
            endpoint_config:     Endpoint configuration.
            secret_manager_name:      Name of the secret manager.

        Returns:
            Credentials from secret manager.
                Example: {"username": "some user", "password": "some password"}
        """
        secret_manager = EndpointResolver._get_secret_manager(secret_manager_name)

        if secret_manager is None:
            raise RasaException(
                f"Failed to load secret manager {secret_manager_name}. "
            )

        endpoint_trait = EndpointTrait(
            endpoint_config=endpoint_config, endpoint_type="tracker_store"
        )
        secrets = secret_manager.load_secrets(endpoint_trait)

        return EndpointResolver._map_secrets_to_endpoint_properties(
            endpoint_properties=endpoint_properties,
            endpoint_config=endpoint_config,
            secrets=secrets,
            secret_manager_name=secret_manager_name,
        )

    @staticmethod
    def _map_secrets_to_endpoint_properties(
        endpoint_properties: List[Text],
        endpoint_config: EndpointConfig,
        secrets: Dict[Text, Text],
        secret_manager_name: Text,
    ) -> Optional[Dict[Text, Text]]:
        """Map secrets to endpoint properties.

        Args:
            endpoint_properties: The endpoint properties to map.
            endpoint_config: The endpoint config to which the secrets belong.
            secrets: The secrets to map.
            secret_manager_name: The name of the secret manager.

        Returns:
            The secrets mapped to their endpoint property.
        """
        credentials: Dict[Text, Text] = {}
        if secrets:
            for endpoint_property_name in endpoint_properties:
                secret_name = EndpointResolver.get_secret_name(
                    endpoint_config, endpoint_property_name
                )
                if secret_name and secret_name in secrets.keys():
                    received_secret = secrets.get(secret_name)

                    if received_secret is not None:
                        credentials[endpoint_property_name] = received_secret
                else:
                    logger.warning(
                        f"Failed to load {endpoint_property_name} "
                        f"from secrets manager {secret_manager_name}."
                    )
        return credentials if credentials else None

    @staticmethod
    def get_secret_name(
        endpoint_config: EndpointConfig, endpoint_property_name: Text
    ) -> Optional[Text]:
        """Get the secret name from the endpoint config.

        Args:
            endpoint_config: Endpoint configuration.
            endpoint_property_name: Endpoint property name.

        Returns:
            Secret name or None if endpoint property
            value is not managed by secret manager.
        """
        endpoint_reader = EndpointReader(endpoint_config=endpoint_config)
        if not endpoint_reader.has_property(endpoint_property_name):
            return None

        property_value = endpoint_reader.get_property_value(
            endpoint_property_name=endpoint_property_name
        )

        if property_value and CredentialsLocation.is_credentials_location_instance(
            property_value
        ):
            credentials_location = cast(CredentialsLocation, property_value)
            return credentials_location.secret_key

        return None

    @staticmethod
    def update_config(
        endpoint_config: EndpointConfig,
    ) -> EndpointConfig:
        """Update the endpoint config with the credentials from the secret manager."""
        credentials = EndpointResolver._get_credentials_from_secret_managers(
            endpoint_config=endpoint_config
        )

        if credentials is None:
            return endpoint_config

        result_endpoint_config = copy.deepcopy(endpoint_config)
        for attribute_name in result_endpoint_config.__dict__.keys():
            if attribute_name in credentials:
                result_endpoint_config.__dict__[attribute_name] = credentials[
                    attribute_name
                ]

        for kwarg_name in result_endpoint_config.kwargs.keys():
            if kwarg_name in credentials:
                result_endpoint_config.kwargs[kwarg_name] = credentials[kwarg_name]

        return result_endpoint_config
