from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field, model_validator
from ruamel import yaml as yaml

from rasa.exceptions import ValidationError
from rasa.utils.singleton import Singleton

DEFAULT_AGENTS_CONFIG_FOLDER = "sub_agents"

structlogger = structlog.get_logger()


class ProtocolConfig(str, Enum):
    """Supported protocols for agents."""

    RASA = "RASA"
    A2A = "A2A"


class AgentInfo(BaseModel):
    """Configuration for an agent."""

    name: str = Field(..., description="Agent name")
    protocol: ProtocolConfig = Field(
        default=ProtocolConfig.RASA,
        description="Protocol used to communicate with the agent.",
    )
    description: str = Field(..., description="Agent description")

    @model_validator(mode="before")
    @classmethod
    def validate_protocol(cls, values: Any) -> Any:
        """Validate and normalize protocol values to support lowercase input."""
        if isinstance(values, dict) and "protocol" in values:
            protocol_value = values["protocol"]
            if isinstance(protocol_value, str):
                # Map lowercase protocol names to uppercase enum values
                protocol_mapping = {
                    "rasa": ProtocolConfig.RASA,
                    "a2a": ProtocolConfig.A2A,
                    "RASA": ProtocolConfig.RASA,
                    "A2A": ProtocolConfig.A2A,
                }

                if protocol_value.lower() in protocol_mapping:
                    values["protocol"] = protocol_mapping[protocol_value.lower()]
                else:
                    # If it's not a recognized protocol, let Pydantic handle the
                    # validation
                    # This will raise a proper validation error
                    pass

        return values


class AgentConfiguration(BaseModel):
    llm: Optional[Dict[str, Any]] = None
    prompt_template: Optional[str] = None
    module: Optional[str] = None
    timeout: Optional[int] = None  # timeout in seconds
    max_retries: Optional[int] = None
    agent_card: Optional[str] = None
    auth: Optional[Dict[str, Any]] = None


class AgentConnections(BaseModel):
    mcp_servers: Optional[List[AgentMCPServerConfig]] = None


class AgentMCPServerConfig(BaseModel):
    name: str  # Reference to MCPServerConfig
    url: Optional[str] = None
    type: Optional[str] = None
    include_tools: Optional[List[str]] = None
    exclude_tools: Optional[List[str]] = None
    # Additional parameters for the MCP server
    additional_params: Optional[Dict[str, Any]] = None


class AgentConfig(BaseModel):
    agent: AgentInfo
    configuration: Optional[AgentConfiguration] = None
    connections: Optional[AgentConnections] = None


class AvailableAgents(metaclass=Singleton):
    """Collection of configured agents."""

    _instance = None

    def __init__(self, agents: Optional[Dict[str, AgentConfig]] = None) -> None:
        """Create an `AvailableAgents` object."""
        self.agents = agents or {}

    @classmethod
    def _read_agent_folder(cls, agent_folder: str) -> AvailableAgents:
        """Read the different agents from the given folder."""
        agents: Dict[str, AgentConfig] = {}

        if not os.path.isdir(agent_folder):
            if agent_folder != DEFAULT_AGENTS_CONFIG_FOLDER:
                # User explicitly specified a folder, it should exist
                structlogger.error(
                    f"The specified agents config folder '{agent_folder}' does not "
                    f"exist or is not a directory."
                )
                raise ValueError(
                    f"The specified agents config folder '{agent_folder}' does not "
                    f"exist or is not a directory."
                )
            else:
                # We are using the default folder, it may not be created yet
                # Init with an empty agents in this case
                structlogger.info(
                    f"Default agents config folder '{agent_folder}' does not exist. "
                    f"Agent configurations won't be loaded."
                )
                return cls(agents)

        # First, load all agent configs into a temporary list for validation
        agent_configs: List[AgentConfig] = []
        for agent_name in os.listdir(agent_folder):
            config_path = os.path.join(agent_folder, agent_name, "config.yml")
            if not os.path.isfile(config_path):
                continue
            try:
                agent_config = cls._read_agent_config(config_path)
                if not isinstance(agent_config, AgentConfig):
                    raise ValueError(f"Invalid agent config type for {agent_name}")
                agent_configs.append(agent_config)
            except Exception as e:
                raise ValidationError(
                    code="agent.load_failed",
                    event_info=f"Failed to load agent '{agent_name}': {e}",
                    details={
                        "agent_name": agent_name,
                        "agent_folder": agent_folder,
                        "error": str(e),
                    },
                )

        # Validate agent names are unique before adding to dictionary
        from rasa.agents.validation import validate_agent_names_unique

        validate_agent_names_unique(agent_configs)

        for agent_config in agent_configs:
            agents[agent_config.agent.name] = agent_config

        return cls(agents)

    @staticmethod
    def _read_agent_config(config_path: str) -> AgentConfig:
        """Read the agent config from a yaml file into Pydantic models.

        Args:
            config_path: Path to the config file.

        Returns:
            The parsed AgentConfig.

        Raises:
            yaml.YAMLError: If the YAML file is invalid.
            ValueError: If the data structure is invalid for Pydantic models.
        """
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        # Create the agent config (this will trigger Pydantic validation)
        agent_config = AgentConfig(
            agent=AgentInfo(**data.get("agent", {})),
            configuration=AgentConfiguration(**data.get("configuration", {}))
            if data.get("configuration")
            else None,
            connections=AgentConnections(**data.get("connections", {}))
            if data.get("connections")
            else None,
        )

        return agent_config

    @classmethod
    def get_instance(
        cls, agent_folder: Optional[str] = DEFAULT_AGENTS_CONFIG_FOLDER
    ) -> AvailableAgents:
        """Get the singleton instance of `AvailableAgents`."""
        if cls._instance is None:
            cls._instance = cls._read_agent_folder(agent_folder)

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None
        # Also clear the metaclass singleton instances
        if hasattr(type(cls), "_instances"):
            type(cls)._instances.clear()

    def as_json_list(self) -> List[Dict[str, Any]]:
        """Convert the available agents to a JSON-serializable list."""
        return [
            {
                "name": agent_name,
                "agent": agent_config.agent.model_dump(),
                "configuration": agent_config.configuration.model_dump()
                if agent_config.configuration
                else None,
                "connections": agent_config.connections.model_dump()
                if agent_config.connections
                else None,
            }
            for agent_name, agent_config in self.agents.items()
        ]

    @classmethod
    def get_agent_config(cls, agent_id: str) -> Optional[AgentConfig]:
        instance = cls.get_instance()
        return instance.agents.get(agent_id)

    @classmethod
    def has_agents(cls) -> bool:
        instance = cls.get_instance()
        return len(instance.agents) > 0
