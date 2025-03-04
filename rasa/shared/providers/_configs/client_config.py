from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClientConfig(Protocol):
    """
    Protocol for the client config that specifies the interface for interacting
    with the API.
    """

    @classmethod
    def from_dict(cls, config: dict) -> ClientConfig:
        """
        Initializes the client config with the given configuration.

        This class method should be implemented to parse the given
        configuration and create an instance of an client config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            ClientConfig
        """
        ...

    def to_dict(self) -> dict:
        """
        Returns the configuration for that the client config is initialized with.

        This method should be implemented to return a dictionary containing
        the configuration settings for the client config.

        Returns:
            dictionary containing the configuration settings for the client config.
        """
        ...

    @staticmethod
    def resolve_config_aliases(config: dict) -> dict:
        """
        Resolve any potential aliases in the configuration.

        This method should be implemented to resolve any potential aliases in the
        configuration.

        Args:
            config: (dict) The config from which to initialize.

        Returns:
            dictionary containing the resolved configuration settings for the
            client config.
        """
        ...
