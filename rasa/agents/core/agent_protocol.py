from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rasa.agents.core.types import ProtocolType
    from rasa.agents.schemas.agent_input import AgentInput
    from rasa.agents.schemas.agent_output import AgentOutput
    from rasa.core.available_agents import AgentConfig
    from rasa.core.channels.channel import OutputChannel


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Python protocol for interfacing with agent clients that implement the
    agent communication protocols like MCP, A2A, ACP, etc.
    """

    @classmethod
    def from_config(cls, config: AgentConfig) -> AgentProtocol:
        """Initialize the Agent with the given configuration.

        This class method should be implemented to parse the given
        configuration and create an instance of an agent.

        Args:
            config: The configuration for the agent as an AgentConfig object.

        Returns:
            An instance of the agent.
        """
        ...

    @property
    def protocol_type(self) -> "ProtocolType":
        """
        Returns the protocol type of the agent.

        This property should be implemented to return the protocol type of the agent.
        This is used to determine the type of agent to create in the AgentFactory.
        """
        ...

    async def connect(self) -> None:
        """
        Establish connection to agent/server.

        This method should be implemented to establish a connection to the agent/server
        and load any necessary resources.
        """
        ...

    async def disconnect(self) -> None:
        """
        Close connection to agent/server.

        This method should be implemented to close the connection to the agent/server
        and release any necessary resources.
        """
        ...

    async def process_input(self, input: "AgentInput") -> "AgentInput":
        """
        Pre-process the input before sending it to the agent.

        This method should be implemented to pre-process the input before sending it
        to the agent. This can be used to add any necessary metadata or context to the
        input, or to filter the input.

        Args:
            input: The input to the agent as an AgentInput object.

        Returns:
            The processed input to the agent as an AgentInput object.
        """
        ...

    async def run(
        self, input: "AgentInput", output_channel: Optional[OutputChannel] = None
    ) -> "AgentOutput":
        """Send a message to Agent/server and return response.

        This method should be implemented to send a message to the agent/server and
        return the response in an AgentOutput object.

        Args:
            input: The input to the agent as an AgentInput object.

        Returns:
            The output from the agent as an AgentOutput object.
        """
        ...

    async def process_output(self, output: "AgentOutput") -> "AgentOutput":
        """
        Post-process the output before returning it to Rasa.

        This method should be implemented to post-process the output before returning
        it to Rasa. This can be used to add any necessary metadata or context to the
        output, or to filter the output.

        Args:
            output: The output from the agent as an AgentOutput object.

        Returns:
            The processed output from the agent as an AgentOutput object.
        """
        ...
