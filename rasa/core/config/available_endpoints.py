from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rasa.core.constants import MCP_SERVERS_KEY
from rasa.shared.agents.auth.utils import validate_secrets_in_params
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


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(..., description="The name of the MCP server.")
    url: str = Field(..., description="The URL of the MCP server.")
    type: str = Field(..., description="The type of the MCP server.")
    additional_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional parameters for the MCP server."
    )

    @model_validator(mode="after")
    def validate_type(self) -> MCPServerConfig:
        # validate that type is "http"
        if self.type not in ["http", "https"]:
            raise ValueError(f"Invalid MCP server type: {self.type}")
        # validate that name and url are not empty
        if not self.name or not self.url:
            raise ValueError("Name and URL cannot be empty")
        # validate secrets in additional_params
        if self.additional_params:
            validate_secrets_in_params(
                self.additional_params, f"MCP server - '{self.name}'"
            )
        return self

    @model_validator(mode="before")
    def collect_additional_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        base_fields = {"name", "url", "type"}
        extras = {k: v for k, v in values.items() if k not in base_fields}
        if extras:
            values["additional_params"] = extras
            # remove them from top level so Pydantic doesn’t complain
            for k in extras:
                values.pop(k)
        return values


class AvailableEndpoints:
    """Collection of configured endpoints."""

    @classmethod
    def read_endpoints(cls, endpoint_file: Path) -> AvailableEndpoints:
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
        raw_mcp_servers = read_property_config_from_endpoints_file(
            endpoint_file, property_name=MCP_SERVERS_KEY
        )
        mcp_servers = (
            [MCPServerConfig(**server) for server in raw_mcp_servers]
            if raw_mcp_servers
            else None
        )
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
            endpoint_file,
            nlg,
            nlu,
            action,
            model,
            tracker_store,
            lock_store,
            event_broker,
            vector_store,
            mcp_servers,
            model_groups,
            privacy,
            interaction_handling,
        )

    def __init__(
        self,
        config_file_path: Optional[Path] = None,
        nlg: Optional[EndpointConfig] = None,
        nlu: Optional[EndpointConfig] = None,
        action: Optional[EndpointConfig] = None,
        model: Optional[EndpointConfig] = None,
        tracker_store: Optional[EndpointConfig] = None,
        lock_store: Optional[EndpointConfig] = None,
        event_broker: Optional[EndpointConfig] = None,
        vector_store: Optional[EndpointConfig] = None,
        mcp_servers: Optional[List[MCPServerConfig]] = None,
        model_groups: Optional[List[Dict[str, Any]]] = None,
        privacy: Optional[Dict[str, Any]] = None,
        interaction_handling: InteractionHandlingConfig = InteractionHandlingConfig(
            global_silence_timeout=GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE
        ),
    ) -> None:
        """Create an `AvailableEndpoints` object."""
        self.config_file_path = config_file_path
        self.model = model
        self.action = action
        self.nlu = nlu
        self.nlg = nlg
        self.tracker_store = tracker_store
        self.lock_store = lock_store
        self.event_broker = event_broker
        self.vector_store = vector_store
        self.mcp_servers = mcp_servers
        self.model_groups = model_groups
        self.privacy = privacy
        self.interaction_handling = interaction_handling
