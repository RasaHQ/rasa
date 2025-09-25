"""Test utilities for MCP agent testing."""

from typing import Any, Dict, List, Optional

from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
from rasa.agents.schemas import AgentInput, AgentOutput
from rasa.core.available_agents import AgentMCPServerConfig, ProtocolConfig


class TestMCPBaseAgentImpl(MCPBaseAgent):
    """Concrete implementation of MCPBaseAgent for testing purposes."""

    def __init__(
        self,
        name: str,
        description: str,
        protocol_type: ProtocolConfig,
        server_configs: List[AgentMCPServerConfig],
        llm_config: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        super().__init__(
            name,
            description,
            protocol_type,
            server_configs,
            llm_config,
            prompt_template,
            timeout,
            max_retries,
        )

    @classmethod
    def get_default_prompt_template(cls) -> str:
        """Return a simple test template."""
        return (
            "Test template: {{user_message}}\nPrevious conversation: "
            "{{conversation_history}}\nCurrent date: {{current_date}} (YYYY-MM-DD)"
            "\nCurrent time: {{current_time}} (HH:MM:SS, 24-hour format)"
            "\nCurrent day: {{current_day}}"
        )

    @property
    def protocol_type(self) -> ProtocolType:
        """Return MCP_OPEN protocol type for testing."""
        return ProtocolType.MCP_OPEN

    async def send_message(self, agent_input: AgentInput) -> AgentOutput:
        """Test implementation of send_message."""
        # Simple test implementation that returns a basic response
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.COMPLETED,
            response_message="Test response",
        )
