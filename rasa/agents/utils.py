from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rasa.core.available_agents import (
    AgentConfig,
    AgentMCPServerConfig,
)
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.shared.core.events import AgentCompleted, AgentStarted
from rasa.shared.core.trackers import DialogueStateTracker

if TYPE_CHECKING:
    from rasa.agents.schemas import AgentInput


def resolve_agent_config(
    agent_config: AgentConfig,
    available_endpoints: AvailableEndpoints,
) -> Optional[AgentConfig]:
    if agent_config is None:
        return None

    connections = agent_config.connections
    mcp_connections: List[AgentMCPServerConfig] = (
        connections.mcp_servers
        if connections and connections.mcp_servers is not None
        else []
    )

    for mcp_server in mcp_connections:
        for mcp_server_endpoint in available_endpoints.mcp_servers or []:
            if mcp_server_endpoint.name == mcp_server.name:
                mcp_server.url = mcp_server_endpoint.url
                mcp_server.type = mcp_server_endpoint.type
                mcp_server.additional_params = mcp_server_endpoint.additional_params

    return agent_config


def is_agent_valid(agent_id: str) -> bool:
    """Check if an agent ID references a valid agent.

    Args:
        agent_id: The agent ID to validate.

    Returns:
        True if the agent exists, False otherwise.
    """
    agent_config = Configuration.get_instance().available_agents.get_agent_config(
        agent_id
    )
    return agent_config is not None


def is_agent_completed(tracker: DialogueStateTracker, agent_id: str) -> bool:
    """Check if an agent has been completed.

    Args:
        tracker: The dialogue state tracker.
        agent_id: The agent ID to check.

    Returns:
        True if the agent has been completed, False otherwise.
    """
    # Look for AgentCompleted events for this agent
    for event in reversed(tracker.events):
        if isinstance(event, AgentCompleted) and event.agent_id == agent_id:
            return True
        elif isinstance(event, AgentStarted) and event.agent_id == agent_id:
            return False
    return False


def get_agent_info(agent_id: str) -> Optional[Dict[str, str]]:
    """Get basic agent information (name and description).

    Args:
        agent_id: The agent ID to get information for.

    Returns:
        Dictionary with agent name and description if found, None otherwise.
    """
    agent_config = Configuration.get_instance().available_agents.get_agent_config(
        agent_id
    )
    if agent_config is None:
        return None

    return {
        "name": agent_config.agent.name,
        "description": agent_config.agent.description,
    }


def get_completed_agents_info(tracker: DialogueStateTracker) -> List[Dict[str, str]]:
    """Get information for all completed agents in the currently active flow.

    Args:
        tracker: The dialogue state tracker.

    Returns:
        List of dictionaries containing agent information for completed agents.
    """
    from rasa.dialogue_understanding.stack.utils import top_user_flow_frame

    # Get the currently active flow
    top_flow_frame = top_user_flow_frame(tracker.stack)
    if not top_flow_frame:
        # No active flow, return empty list
        return []

    current_flow_id = top_flow_frame.flow_id
    completed_agents = []

    # Get all agents that completed in the current flow
    agents_completed_in_current_flow = set()
    for event in reversed(tracker.events):
        if isinstance(event, AgentCompleted) and event.flow_id == current_flow_id:
            agents_completed_in_current_flow.add(event.agent_id)

    # Only include agents that are completed (not currently running)
    for agent_id in agents_completed_in_current_flow:
        if is_agent_completed(tracker, agent_id):
            agent_info = get_agent_info(agent_id)
            if agent_info:
                completed_agents.append(agent_info)

    return completed_agents


def get_active_agent_info(
    tracker: DialogueStateTracker, flow_id: str
) -> Optional[Dict[str, str]]:
    """Get information for the active agent in a specific flow.

    Args:
        tracker: The dialogue state tracker.
        flow_id: The flow ID to get the active agent for.

    Returns:
        Dictionary with agent name and description if an agent is active,
        None otherwise.
    """
    agent_frame = tracker.stack.find_active_agent_stack_frame_for_flow(flow_id)
    if agent_frame:
        return get_agent_info(agent_frame.agent_id)
    return None


def get_slot_value_from_agent_input(
    context: "AgentInput", slot_name: str
) -> Optional[Any]:
    """Get the slot value from the agent input."""
    for slot in context.slots:
        if slot.name == slot_name and slot.value is not None:
            return slot.value
    return None
