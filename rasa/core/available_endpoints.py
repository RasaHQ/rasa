from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Union

from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH
from rasa.shared.core.constants import (
    GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE,
    GLOBAL_SILENCE_TIMEOUT_KEY,
)
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import (
    EndpointConfig,
    read_endpoint_config,
    read_property_config_from_endpoints_file,
)


@dataclasses.dataclass
class InteractionHandlingConfig:
    """Configuration for interaction handling."""

    global_silence_timeout: Union[float, int] = GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE

    def __post_init__(self) -> None:
        # Validate the type of `global_silence_timeout`.
        if isinstance(self.global_silence_timeout, str):
            try:
                self.global_silence_timeout = float(self.global_silence_timeout)
            except ValueError:
                raise RasaException(
                    f"Type for {GLOBAL_SILENCE_TIMEOUT_KEY} is wrong, expected number. "
                    f"Got: '{self.global_silence_timeout}'. "
                )

        if not isinstance(self.global_silence_timeout, (float, int)):
            raise RasaException(
                f"Type for {GLOBAL_SILENCE_TIMEOUT_KEY} is wrong, expected number. "
                f"Got: '{type(self.global_silence_timeout)}'. "
            )

        if self.global_silence_timeout <= 0:
            raise RasaException(
                f"Value for {GLOBAL_SILENCE_TIMEOUT_KEY} must be a positive number. "
                f"Got: '{self.global_silence_timeout}'. "
            )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> InteractionHandlingConfig:
        """Create a InteractionHandlingConfig instance from a dictionary."""
        return cls(
            global_silence_timeout=data.get(
                GLOBAL_SILENCE_TIMEOUT_KEY, GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE
            )
            if data is not None
            else GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE
        )


class AvailableEndpoints:
    """Collection of configured endpoints."""

    _instance = None

    @classmethod
    def read_endpoints(cls, endpoint_file: str) -> AvailableEndpoints:
        """Read the different endpoints from a yaml file."""
        nlg = read_endpoint_config(endpoint_file, endpoint_type="nlg")
        nlu = read_endpoint_config(endpoint_file, endpoint_type="nlu")
        action = read_endpoint_config(endpoint_file, endpoint_type="action_endpoint")
        model = read_endpoint_config(endpoint_file, endpoint_type="models")
        tracker_store = read_endpoint_config(
            endpoint_file, endpoint_type="tracker_store"
        )
        lock_store = read_endpoint_config(endpoint_file, endpoint_type="lock_store")
        event_broker = read_endpoint_config(endpoint_file, endpoint_type="event_broker")
        vector_store = read_endpoint_config(endpoint_file, endpoint_type="vector_store")
        model_groups = read_property_config_from_endpoints_file(
            endpoint_file, property_name="model_groups"
        )
        privacy = read_property_config_from_endpoints_file(
            endpoint_file, property_name="privacy"
        )

        interaction_handling = InteractionHandlingConfig.from_dict(
            read_property_config_from_endpoints_file(
                endpoint_file, property_name="interaction_handling"
            )
        )

        return cls(
            nlg,
            nlu,
            action,
            model,
            tracker_store,
            lock_store,
            event_broker,
            vector_store,
            model_groups,
            privacy,
            interaction_handling,
        )

    def __init__(
        self,
        nlg: Optional[EndpointConfig] = None,
        nlu: Optional[EndpointConfig] = None,
        action: Optional[EndpointConfig] = None,
        model: Optional[EndpointConfig] = None,
        tracker_store: Optional[EndpointConfig] = None,
        lock_store: Optional[EndpointConfig] = None,
        event_broker: Optional[EndpointConfig] = None,
        vector_store: Optional[EndpointConfig] = None,
        model_groups: Optional[List[Dict[str, Any]]] = None,
        privacy: Optional[Dict[str, Any]] = None,
        interaction_handling: InteractionHandlingConfig = InteractionHandlingConfig(
            global_silence_timeout=GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE
        ),
    ) -> None:
        """Create an `AvailableEndpoints` object."""
        self.model = model
        self.action = action
        self.nlu = nlu
        self.nlg = nlg
        self.tracker_store = tracker_store
        self.lock_store = lock_store
        self.event_broker = event_broker
        self.vector_store = vector_store
        self.model_groups = model_groups
        self.privacy = privacy
        self.interaction_handling = interaction_handling

    @classmethod
    def get_instance(
        cls, endpoint_file: Optional[str] = DEFAULT_ENDPOINTS_PATH
    ) -> AvailableEndpoints:
        """Get the singleton instance of AvailableEndpoints."""
        # Ensure that the instance is initialized only once.
        if cls._instance is None:
            cls._instance = cls.read_endpoints(endpoint_file)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None
