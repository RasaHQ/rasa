from typing import Optional

from rasa.agents.core.types import AgentIdentifier, ProtocolType
from rasa.core.available_agents import AgentConfig, ProtocolConfig
from rasa.shared.core.flows.steps import CallFlowStep


def make_agent_identifier(
    agent_name: str, protocol_type: ProtocolType
) -> AgentIdentifier:
    """Make an agent identifier."""
    return AgentIdentifier(agent_name, protocol_type)


def get_protocol_type(
    step: CallFlowStep, agent_config: Optional[AgentConfig]
) -> ProtocolType:
    """Get the protocol type for an agent.

    Args:
        step: The step that is calling the agent.
        agent_config: The agent configuration.

    Returns:
        The protocol type for the agent.
    """
    if step.exit_if:
        protocol_type = ProtocolType.MCP_TASK
    else:
        protocol_type = (
            ProtocolType.A2A
            if agent_config and agent_config.agent.protocol == ProtocolConfig.A2A
            else ProtocolType.MCP_OPEN
        )
    return protocol_type
