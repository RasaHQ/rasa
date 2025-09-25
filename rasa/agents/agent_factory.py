from typing import Any, ClassVar, Dict, List, Type

import structlog

from rasa.agents.core.agent_protocol import AgentProtocol
from rasa.agents.core.types import ProtocolType
from rasa.agents.protocol import A2AAgent, MCPOpenAgent, MCPTaskAgent
from rasa.core.available_agents import AgentConfig
from rasa.shared.utils.common import class_from_module_path

structlogger = structlog.get_logger()


class AgentFactory:
    """Factory for creating agent instances based on the protocol type."""

    _protocols: ClassVar[Dict[ProtocolType, Type[AgentProtocol]]] = {
        ProtocolType.A2A: A2AAgent,
        ProtocolType.MCP_OPEN: MCPOpenAgent,
        ProtocolType.MCP_TASK: MCPTaskAgent,
    }

    @classmethod
    def create_client(
        cls, protocol_type: ProtocolType, config: AgentConfig
    ) -> AgentProtocol:
        """Create an agent instance based on the protocol type.

        Args:
            protocol_type: The protocol type of the agent.
            config: The configuration for the agent as an AgentConfig object.

        Returns:
            An instance of the agent.
        """
        # If the agent is a custom agent, we need to create it from the module
        if config.configuration and config.configuration.module:
            agent_class: Type[AgentProtocol] = class_from_module_path(
                config.configuration.module
            )
            if not cls._is_valid_custom_agent(agent_class, protocol_type):
                raise ValueError(
                    f"Agent class `{agent_class}` does not subclass the "
                    f"{cls._get_agent_class_from_protocol(protocol_type).__name__} "
                    f"agent class."
                )
            structlogger.debug(
                "agent_factory.create_client.custom_agent",
                event_info=(
                    f"Initializing `{agent_class.__name__}` for agent "
                    f"`{config.agent.name}` with protocol `{protocol_type.value}`"
                ),
                agent_name=config.agent.name,
                protocol_type=protocol_type.value,
                agent_class=agent_class.__name__,
            )
            return agent_class.from_config(config)

        # If the agent is a built-in agent, we need to create it from the protocol class
        protocol_class = cls._get_agent_class_from_protocol(protocol_type)
        if protocol_class is None:
            raise ValueError(
                f"Unsupported protocol: {protocol_type}. "
                f"Supported protocols: {cls.get_supported_protocols()}"
            )

        structlogger.debug(
            "agent_factory.create_client",
            event_info=(
                f"Initializing `{protocol_class.__name__}` for agent "
                f"`{config.agent.name}` with protocol `{protocol_type.value}`"
            ),
            agent_name=config.agent.name,
            protocol_type=protocol_type.value,
            agent_class=protocol_class.__name__,
        )
        return protocol_class.from_config(config)

    @classmethod
    def register_protocol(
        cls, protocol_type: ProtocolType, protocol_class: Type[AgentProtocol]
    ) -> None:
        """Register new protocol implementation.

        Args:
            protocol_type: The protocol type of the agent.
            protocol_class: The class that implements the protocol.
        """
        if cls.is_protocol_supported(protocol_type):
            raise ValueError(f"Protocol {protocol_type.name} already registered.")
        cls._protocols[protocol_type] = protocol_class

    @classmethod
    def get_supported_protocols(cls) -> List[ProtocolType]:
        """Get all supported protocol types."""
        return list(cls._protocols.keys())

    @classmethod
    def _get_agent_class_from_protocol(
        cls, protocol_type: ProtocolType
    ) -> Type[AgentProtocol]:
        """Get the class that implements the protocol."""
        if not cls.is_protocol_supported(protocol_type):
            raise ValueError(
                f"Unsupported protocol: {protocol_type}. "
                f"Supported protocols: {cls.get_supported_protocols()}"
            )
        return cls._protocols[protocol_type]

    @classmethod
    def is_protocol_supported(cls, protocol_type: ProtocolType) -> bool:
        """Check if the protocol is supported."""
        return protocol_type in cls._protocols

    @classmethod
    def _is_valid_custom_agent(
        cls, agent_class: Any, protocol_type: ProtocolType
    ) -> bool:
        """Check if the agent class is valid."""
        return issubclass(
            agent_class, cls._get_agent_class_from_protocol(protocol_type)
        )
