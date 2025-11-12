from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field, model_validator
from ruamel import yaml as yaml

from rasa.exceptions import ValidationError
from rasa.shared.utils.yaml import read_config_file

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
    include_date_time: Optional[bool] = None
    timezone: Optional[str] = None


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


class AvailableAgents:
    """Collection of configured agents."""

    def __init__(self, agents: Optional[Dict[str, AgentConfig]] = None) -> None:
        """Create an `AvailableAgents` object."""
        self.agents: Dict[str, AgentConfig] = agents or {}

    @classmethod
    def read_from_folder(cls, sub_agents_folder: str) -> AvailableAgents:
        """Read the different agents from the given folder."""
        agents: Dict[str, AgentConfig] = {}

        if not os.path.isdir(sub_agents_folder):
            if sub_agents_folder != DEFAULT_AGENTS_CONFIG_FOLDER:
                # User explicitly specified a folder, it should exist
                raise ValidationError(
                    code="agent.sub_agents_folder_not_found",
                    event_info=f"The specified agents config folder "
                    f"'{sub_agents_folder}' does not exist or is not a "
                    f"directory.",
                    details={"folder": sub_agents_folder},
                )
            else:
                # We are using the default folder, it may not be created yet
                # Init with an empty agents in this case
                structlogger.debug(
                    f"Default agents config folder '{sub_agents_folder}' does not "
                    f"exist. Agent configurations won't be loaded."
                )
                return cls(agents)

        # First, load all agent configs into a temporary list for validation
        agent_configs: List[AgentConfig] = []
        for agent_name in os.listdir(sub_agents_folder):
            agent_folder = os.path.join(sub_agents_folder, agent_name)
            if not os.path.isdir(agent_folder):
                raise ValidationError(
                    code="agent.invalid_directory_structure",
                    event_info=f"Invalid structure: '{agent_folder}' is not a folder. "
                    f"Each agent must be stored in its own folder inside "
                    f"'{sub_agents_folder}'. Expected structure: "
                    f"{sub_agents_folder}/<agent_name>/config.yml",
                    details={
                        "agent_name": agent_name,
                        "sub_agents_folder": sub_agents_folder,
                    },
                )
            config_path = os.path.join(agent_folder, "config.yml")
            if not os.path.isfile(config_path):
                raise ValidationError(
                    code="agent.missing_config_file",
                    event_info=f"Missing config file for agent '{agent_name}'. "
                    f"Expected file: '{config_path}'. "
                    f"Each agent folder must contain a 'config.yml' file.",
                    details={
                        "agent_name": agent_name,
                        "expected_config_file": config_path,
                        "sub_agents_folder": sub_agents_folder,
                    },
                )
            try:
                agent_config = cls._read_agent_config_file(config_path)
                if not isinstance(agent_config, AgentConfig):
                    raise ValueError(f"Invalid agent config type for {agent_name}")
                agent_configs.append(agent_config)
            except Exception as e:
                raise ValidationError(
                    code="agent.load_failed",
                    event_info=f"Failed to load agent '{agent_name}': {e}",
                    details={
                        "agent_name": agent_name,
                        "sub_agents_folder": sub_agents_folder,
                        "error": str(e),
                    },
                )

        # Validate agent names are unique before adding to dictionary
        from rasa.agents.validation import validate_agent_names_unique

        validate_agent_names_unique(agent_configs)

        for agent_config in agent_configs:
            agents[agent_config.agent.name] = agent_config

        structlogger.info(f"Loaded agent configs: {[k for k in agents.keys()]}")
        return cls(agents)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> AgentConfig:
        """Parse the agent config from raw data into Pydantic models.

        Args:
            data: Raw data from the config file as a dictionary.

        Returns:
            The parsed AgentConfig as a Pydantic model.

        Raises:
            ValueError: If the data structure is invalid for Pydantic models.
        """
        return AgentConfig(
            agent=AgentInfo(**data.get("agent", {})),
            configuration=AgentConfiguration(**data.get("configuration", {}))
            if data.get("configuration")
            else None,
            connections=AgentConnections(**data.get("connections", {}))
            if data.get("connections")
            else None,
        )

    @classmethod
    def _read_agent_config_file(cls, config_path: str) -> AgentConfig:
        """Read the agent config from a yaml file into Pydantic models.

        Args:
            config_path: Path to the config file.

        Returns:
            The parsed AgentConfig.

        Raises:
            yaml.YAMLError: If the YAML file is invalid.
            ValidationError: If the data structure is invalid for Pydantic models.
        """
        data = read_config_file(config_path)
        return cls.from_dict(data)

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

    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        return self.agents.get(agent_id)

    def has_agents(self) -> bool:
        return len(self.agents) > 0
