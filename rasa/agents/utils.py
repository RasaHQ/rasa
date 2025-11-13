from typing import Dict, List, Optional, Type

import structlog

from rasa.agents.agent_manager import AgentManager
from rasa.agents.exceptions import AgentNotFoundException
from rasa.agents.validation import validate_agent_names_not_conflicting_with_flows
from rasa.core.available_agents import (
    AgentConfig,
    AgentMCPServerConfig,
    AvailableAgents,
)
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.shared.agents.utils import get_protocol_type
from rasa.shared.core.events import AgentCompleted, AgentStarted
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps import CallFlowStep
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


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


async def initialize_agents(
    flows: FlowsList,
    sub_agents: AvailableAgents,
) -> None:
    """Iterate over flows and create/connect the referenced agents."""
    agent_manager: AgentManager = AgentManager()
    endpoints = Configuration.get_instance().endpoints

    agent_used = False
    mcp_tool_used = False

    # Validate agent names don't conflict with flow names
    flow_names = {flow.id for flow in flows.underlying_flows}
    validate_agent_names_not_conflicting_with_flows(sub_agents.agents, flow_names)

    for flow in flows.underlying_flows:
        for step in flow.steps:
            if isinstance(step, CallFlowStep):
                if flows.flow_by_id(step.call) is not None:
                    continue

                if step.is_calling_mcp_tool():
                    # The call step is calling an MCP tool, so we don't need to
                    # initialize any agent.
                    mcp_tool_used = True
                    continue

                if not step.is_calling_agent():
                    raise AgentNotFoundException(step.call)

                agent_used = True

                agent_name = step.call
                agent_config = sub_agents.get_agent_config(agent_name)
                resolved_agent_config = resolve_agent_config(agent_config, endpoints)
                protocol_type = get_protocol_type(step, agent_config)

                await agent_manager.connect_agent(
                    agent_name,
                    protocol_type,
                    resolved_agent_config,
                )

    _log_beta_feature_warning(mcp_tool_used, agent_used)


def _log_beta_feature_warning(mcp_tool_used: bool, agent_used: bool) -> None:
    """Log a warning if an agent or MCP tool is used in the flow(s)."""
    if mcp_tool_used:
        structlogger.info(
            "rasa.agents.utils.initialize_agents",
            event_info="Beta Feature",
            message=(
                "An MCP tool is being called in at least one of the flows. "
                "This feature is currently under beta development. "
                "It might undergo upgrades / breaking changes in future "
                "releases to graduate it to GA."
            ),
        )

    if agent_used:
        structlogger.info(
            "rasa.agents.utils.initialize_agents",
            event_info="Beta Feature",
            message=(
                "A sub-agent is being called from a flow. "
                "This feature is currently under beta development. "
                "It might undergo upgrades / breaking changes in future "
                "releases to graduate it to GA."
            ),
        )


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


class AgentsConnectionCleanup:
    """Context manager that ensures proper cleanup of agent connections.

    This is cleanup-only RAII - agents connections are not acquired here,
    but are cleaned up when the context exits.

    Usage:
        async with AgentsConnectionCleanup():
            # Agents connections are available here
            await some_operation_using_agents()
        # Agents connections are automatically cleaned up here
    """

    async def __aenter__(self) -> "AgentsConnectionCleanup":
        """Enter the context"""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> bool:
        """Exit the context - cleanup agents connections."""
        agent_manager: AgentManager = AgentManager()
        agents_to_disconnect = agent_manager.agents.keys()
        if agents_to_disconnect:
            structlogger.debug(
                "agents.utils.agents_connection_cleanup.cleanup_starting",
                agent_count=len(agents_to_disconnect),
                event_info=f"Starting cleanup of {len(agents_to_disconnect)} agents",
            )

            # Disconnect each agent using the agent manager.
            for agent_identifier in agents_to_disconnect:
                await agent_manager.disconnect_agent(
                    agent_identifier.agent_name, agent_identifier.protocol_type
                )

            structlogger.debug(
                "agents.utils.agents_connection_cleanup.cleanup_completed",
                event_info="Agents connection cleanup completed",
            )
        return True
