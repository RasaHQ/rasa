from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Text

from rasa.core.config.configuration import Configuration
from rasa.shared.core.flows.flow_step import FlowStep

if TYPE_CHECKING:
    from rasa.shared.core.flows.flow import Flow


@dataclass
class CallFlowStep(FlowStep):
    """Represents the configuration of a call flow/agent step."""

    call: Text
    """The flow to be called or the ID of the agent to be called."""
    called_flow_reference: Optional["Flow"] = None

    # MCP Tool calling
    """The MCP server that hosts the tool to be called."""
    mcp_server: Optional[str] = None
    """The input and output mapping for the MCP tool."""
    mapping: Optional[Dict[str, Any]] = None

    # Call agent exit condition
    """A list of slot predicates that determine when to exit the agent loop."""
    exit_if: Optional[List[str]] = None

    @classmethod
    def from_json(cls, flow_id: Text, data: Dict[Text, Any]) -> CallFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_id: The id of the flow that contains the step.
            data: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super().from_json(flow_id, data)
        return CallFlowStep(
            call=data.get("call", ""),
            mcp_server=data.get("mcp_server", None),
            mapping=data.get("mapping", None),
            exit_if=data.get("exit_if", None),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:  # type: ignore[override]
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        step_properties: Dict[str, Any] = {
            "call": self.call,
        }
        if self.is_calling_mcp_tool():
            step_properties["mcp_server"] = self.mcp_server
            step_properties["mapping"] = self.mapping
        if self.exit_if:
            step_properties["exit_if"] = self.exit_if

        return super().as_json(step_properties=step_properties)

    def steps_in_tree(
        self, should_resolve_calls: bool = True
    ) -> Generator[FlowStep, None, None]:
        """Returns the steps in the tree of the flow step."""
        yield self

        if should_resolve_calls and self.is_calling_flow():
            if not self.called_flow_reference:
                raise ValueError(
                    f"Flow step '{self.id}' in flow '{self.flow_id}' is trying "
                    f"to call flow '{self.call}', but the flow reference could "
                    f"not be resolved. Please ensure that:\n"
                    f"1. A flow named '{self.call}' is defined in your domain\n"
                    f"2. The flow name is spelled correctly (case-sensitive)\n"
                    f"3. The called flow is properly formatted with valid YAML syntax"
                )

            yield from self.called_flow_reference.steps_with_calls_resolved

        yield from self.next.steps_in_tree(should_resolve_calls)

    def is_calling_flow(self) -> bool:
        """Returns True if the call references a flow."""
        return not self.is_calling_mcp_tool() and not self.is_calling_agent()

    def is_calling_mcp_tool(self) -> bool:
        """Returns True if the call references an MCP tool of an existing MCP server."""
        from rasa.shared.utils.mcp.utils import mcp_server_exists

        return self.has_mcp_tool_properties() and mcp_server_exists(self.mcp_server)

    def has_mcp_tool_properties(self) -> bool:
        """Returns True if the call step has MCP tool properties."""
        return self.mcp_server is not None and self.mapping is not None

    def is_calling_agent(self) -> bool:
        """Returns True if the call references an agent."""
        return self.call in Configuration.get_instance().available_agents.agents

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return f"call_{self.call}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return (
                self.call == other.call
                and self.called_flow_reference == other.called_flow_reference
                and self.mcp_server == other.mcp_server
                and self.mapping == other.mapping
                and self.exit_if == other.exit_if
                and super().__eq__(other)
            )
        return False
