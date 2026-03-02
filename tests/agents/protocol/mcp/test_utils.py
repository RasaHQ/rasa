"""Test utilities for MCP agent testing."""

from typing import Any, Dict, List, Optional

from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
from rasa.agents.schemas import AgentInput, AgentOutput
from rasa.core.available_agents import AgentMCPServerConfig, ProtocolConfig
from rasa.core.channels import OutputChannel


class MockMCPBaseAgentImpl(MCPBaseAgent):
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
        include_date_time: Optional[bool] = None,
        timezone: Optional[str] = None,
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
            include_date_time,
            timezone,
        )

    @classmethod
    def get_default_prompt_template(cls) -> str:
        """Return a simple test template."""
        return (
            "Test template: {{user_message}}\nPrevious conversation: "
            "{{conversation_history}}"
            "{% if resumed_after_interruption %}\nResume: {{ resumed_last_request }}"
            "{% endif %}{% if current_datetime %}"
            "- Current date: {{ current_datetime.strftime('%d %B, %Y') }}"
            "- Current time: {{ current_datetime.strftime('%H:%M:%S') }} "
            "({{ current_datetime.tzname() }})"
            "- Current day: {{ current_datetime.strftime('%A') }} {% endif %}"
        )

    @property
    def protocol_type(self) -> ProtocolType:
        """Return MCP_OPEN protocol type for testing."""
        return ProtocolType.MCP_OPEN

    async def send_message(
        self, agent_input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Test implementation of send_message."""
        # Simple test implementation that returns a basic response
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.COMPLETED,
            response_message="Test response",
        )
