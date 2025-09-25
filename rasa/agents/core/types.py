"""Fundamental types for the agent protocol."""

from enum import Enum


class ProtocolType(Enum):
    """An Enum class that represents the supported protocol types."""

    MCP_OPEN = "mcp_open"
    MCP_TASK = "mcp_task"
    A2A = "a2a"


class AgentIdentifier:
    """Represents a unique agent identifier combining agent_id and protocol_type."""

    SEPARATOR = "::"

    def __init__(self, agent_name: str, protocol_type: ProtocolType):
        self.agent_name = agent_name
        self.protocol_type = protocol_type

    def __str__(self) -> str:
        return f"{self.agent_name}{self.SEPARATOR}{self.protocol_type.value}"

    def __repr__(self) -> str:
        return (
            f"AgentIdentifier(agent_name='{self.agent_name}', "
            f"protocol_type='{self.protocol_type.value}')"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AgentIdentifier):
            return (
                self.agent_name == other.agent_name
                and self.protocol_type == other.protocol_type
            )
        elif isinstance(other, str):
            return str(self) == other
        return False

    def __hash__(self) -> int:
        return hash((self.agent_name, self.protocol_type))

    @classmethod
    def from_string(cls, identifier: str) -> "AgentIdentifier":
        """Create an AgentIdentifier from a string representation."""
        if cls.SEPARATOR not in identifier:
            raise ValueError(f"Invalid agent identifier format: {identifier}")

        parts = identifier.split(cls.SEPARATOR, 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid agent identifier format: {identifier}")

        agent_name, protocol_str = parts
        try:
            protocol_type = ProtocolType(protocol_str)
        except ValueError:
            raise ValueError(f"Invalid protocol type: {protocol_str}")

        return cls(agent_name, protocol_type)


class AgentStatus(Enum):
    """A Enum class that represents the status of the agent."""

    COMPLETED = "completed"
    INPUT_REQUIRED = "input_required"
    FATAL_ERROR = "fatal_error"
    RECOVERABLE_ERROR = "recoverable_error"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"AgentStatus.{self.value}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AgentStatus):
            return self.value == other.value
        return False
