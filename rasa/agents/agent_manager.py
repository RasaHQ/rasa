from typing import ClassVar, Dict, Optional

import structlog

from rasa.agents.agent_factory import AgentFactory
from rasa.agents.core.agent_protocol import AgentProtocol
from rasa.agents.core.types import AgentIdentifier, ProtocolType
from rasa.agents.schemas import AgentInput, AgentOutput
from rasa.core.available_agents import AgentConfig
from rasa.core.channels.channel import OutputChannel
from rasa.shared.agents.utils import make_agent_identifier
from rasa.shared.exceptions import AgentInitializationException
from rasa.utils.singleton import Singleton

structlogger = structlog.get_logger()


class AgentManager(metaclass=Singleton):
    """High-level agent management with protocol abstraction."""

    agents: ClassVar[Dict[AgentIdentifier, AgentProtocol]] = {}

    def _add_agent(
        self, agent_identifier: AgentIdentifier, agent: AgentProtocol
    ) -> None:
        """Add an agent to the manager.

        Args:
            agent_identifier: The identifier of the agent.
            agent: The agent to add.
        """
        if agent_identifier in self.agents:
            raise ValueError(f"Agent {agent_identifier} already exists")
        self.agents[agent_identifier] = agent

    def get_agent(self, agent_name: str, protocol_type: ProtocolType) -> AgentProtocol:
        """Retrieve connected agent instance.

        Args:
            agent_name: The name of the agent.
            protocol_type: The protocol type of the agent.

        Returns:
            The agent.

        Raises:
            ValueError: If the agent is not connected.
        """
        agent_identifier = make_agent_identifier(agent_name, protocol_type)
        if agent_identifier not in self.agents:
            raise ValueError(f"Agent {agent_identifier} is not available")
        return self.agents[agent_identifier]

    async def connect_agent(
        self, agent_name: str, protocol_type: ProtocolType, config: AgentConfig
    ) -> None:
        """Connect to agent using specified protocol.

        Also, load the default resources and persist the agent to the manager
        in a ready-to-use state so that it can be used immediately
        to send messages to the agent.

        Args:
            agent_name: The name of the agent.
            protocol_type: The protocol type of the agent.
            config: The configuration for the agent as an AgentConfig object.

        Raises:
            ConnectionError: If the agent connection fails.
        """
        # Add the agent to the manager
        agent_identifier = make_agent_identifier(agent_name, protocol_type)
        if agent_identifier in self.agents:
            structlogger.info(
                "agent_manager.connect_agent.already_connected",
                agent_id=str(agent_identifier),
                agent_name=agent_name,
                event_info=f"Agent {agent_identifier} already connected",
            )
            return
        try:
            # Create the agent client
            client = AgentFactory.create_client(protocol_type, config)

            # Connect the agent client
            await client.connect()

            self._add_agent(agent_identifier, client)
            structlogger.info(
                "agent_manager.connect_agent.success",
                agent_id=str(agent_identifier),
                agent_name=agent_name,
                event_info=f"Connected to agent - `{agent_identifier}` successfully",
            )
        except Exception as e:
            event_info = f"Failed to connect agent {agent_identifier}"
            structlogger.error(
                "agent_manager.connect_agent.failed_to_connect",
                agent_id=str(agent_identifier),
                event_info=event_info,
            )
            raise AgentInitializationException(e, suppress_stack_trace=True) from e

    async def run_agent(
        self,
        agent_name: str,
        protocol_type: ProtocolType,
        context: AgentInput,
        output_channel: Optional[OutputChannel] = None,
    ) -> AgentOutput:
        """Run an agent, send the input to the agent and return the agent response.

        Args:
            agent_name: The name of the agent.
            protocol_type: The protocol type of the agent.
            context: The input to the agent as an AgentInput object.
            output_channel: The output channel to use.

        Returns:
            The response from the agent.
        """
        agent = self.get_agent(agent_name, protocol_type)

        structlogger.debug(
            "agent_manager.run_agent.input",
            event_info="Processing agent input before sending...",
            agent_name=agent_name,
            protocol_type=protocol_type,
        )

        # Process input before sending
        try:
            processed_input = await agent.process_input(context)
        except Exception as e:
            structlogger.error(
                "agent_manager.run_agent.process_input_failed",
                agent_name=agent_name,
                protocol_type=protocol_type,
                event_info=(
                    f"Failed to process input for agent '{agent_name}'. "
                    "Please check your custom implementation."
                ),
                error_message=str(e),
            )
            raise

        # Send message to agent
        output = await agent.run(processed_input, output_channel=output_channel)

        structlogger.debug(
            "agent_manager.run_agent.output",
            event_info="Agent output received. Processing output...",
            agent_name=agent_name,
            protocol_type=protocol_type,
        )

        # Process output before returning
        try:
            processed_output = await agent.process_output(output)
        except Exception as e:
            structlogger.error(
                "agent_manager.run_agent.process_output_failed",
                agent_name=agent_name,
                protocol_type=protocol_type,
                event_info=(
                    f"Failed to process output for agent '{agent_name}'. "
                    "Please check your custom implementation."
                ),
                error_message=str(e),
            )
            raise

        return processed_output

    async def disconnect_agent(
        self, agent_name: str, protocol_type: ProtocolType
    ) -> None:
        """Disconnect agent - Gracefully exit network connections.

        Args:
            agent_name: The name of the agent.
            protocol_type: The protocol type of the agent.

        Raises:
            ValueError: If the agent is not available.
            ConnectionError: If the agent disconnection fails.
        """
        agent_identifier = make_agent_identifier(agent_name, protocol_type)
        if agent_identifier not in self.agents:
            raise ValueError(f"Agent `{agent_identifier}` is not available")
        try:
            await self.get_agent(agent_name, protocol_type).disconnect()

            structlogger.debug(
                "agent_manager.disconnect_agent.success",
                agent_id=str(agent_identifier),
                event_info=f"Disconnected from agent `{agent_identifier}` successfully",
            )
        except Exception as e:
            event_info = f"Failed to disconnect agent `{agent_identifier}`"
            structlogger.error(
                "agent_manager.disconnect_agent.failed_to_disconnect",
                agent_id=str(agent_identifier),
                event_info=event_info,
            )
            raise ConnectionError(e) from e
