from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class RouterClient(Protocol):
    """
    Protocol for a Router client that specifies the interface for interacting
    with the API.
    """

    @classmethod
    def from_config(cls, config: dict) -> RouterClient:
        """
        Initializes the router client with the given configuration.

        This class method should be implemented to parse the given
        configuration and create an instance of an router client.
        """
        ...

    @property
    def config(self) -> Dict:
        """
        Returns the configuration for that the router client is initialized with.

        This property should be implemented to return a dictionary containing
        the client configuration settings for the router client.
        """
        ...

    @property
    def router_settings(self) -> Dict[str, Any]:
        """
        Returns the router settings for the Router client.

        This property should be implemented to return a dictionary containing
        the router settings for the router client.
        """
        ...

    @property
    def model_group_id(self) -> str:
        """
        Returns the model group ID for the Router client.

        This property should be implemented to return the model group ID
        for the router client.
        """
        ...

    @property
    def model_configurations(self) -> List[Dict[str, Any]]:
        """
        Returns the list of model configurations for the Router client.

        This property should be implemented to return the list of model configurations
        for the router client as a list of dictionaries.

        Each dictionary should contain the model configuration.
        Ideally, the `ModelGroupConfig` should parse the model configurations
        and generate this list of dictionaries.
        """
        ...

    @property
    def router_client(self) -> object:
        """
        Returns the instantiated Router client.

        This property should be implemented to return the instantiated
        Router client.
        """
        ...
