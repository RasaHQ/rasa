"""Agent setup utilities for connecting agents referenced in flows."""

from typing import Optional, Type

import structlog

from rasa.agents.agent_manager import AgentManager
from rasa.agents.exceptions import AgentNotFoundException
from rasa.agents.utils import resolve_agent_config
from rasa.agents.validation import validate_agent_names_not_conflicting_with_flows
from rasa.core.available_agents import AvailableAgents
from rasa.core.config.configuration import Configuration
from rasa.shared.agents.utils import get_protocol_type
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps import CallFlowStep

structlogger = structlog.get_logger()


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


def _log_beta_feature_warning(mcp_tool_used: bool, agent_used: bool) -> None:
    """Log a warning if an agent or MCP tool is used in the flow(s)."""
    if mcp_tool_used:
        structlogger.info(
            "rasa.shared.agents.agent_setup.initialize_agents",
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
            "rasa.shared.agents.agent_setup.initialize_agents",
            event_info="Beta Feature",
            message=(
                "A sub-agent is being called from a flow. "
                "This feature is currently under beta development. "
                "It might undergo upgrades / breaking changes in future "
                "releases to graduate it to GA."
            ),
        )
