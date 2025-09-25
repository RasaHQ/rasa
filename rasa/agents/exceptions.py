from rasa.exceptions import RasaException, ValidationError


class AgentNotFoundException(RasaException):
    """Raised when an agent is not found."""

    def __init__(self, agent_name: str):
        super().__init__(f"The agent {agent_name} is not defined.")


class DuplicatedAgentNameException(ValidationError):
    """Raised when agent names are duplicated."""

    def __init__(self, duplicated_names: list[str]) -> None:
        """Initialize the exception."""
        event_info = f"Agent names are duplicated: {', '.join(duplicated_names)}"

        super().__init__(
            code="agent.duplicated_name",
            event_info=event_info,
            duplicated_names=duplicated_names,
        )


class AgentNameFlowConflictException(ValidationError):
    """Raised when agent names conflict with flow names."""

    def __init__(self, conflicting_names: list[str]) -> None:
        """Initialize the exception."""
        event_info = (
            f"Agent names conflict with flow names: {', '.join(conflicting_names)}"
        )

        super().__init__(
            code="agent.flow_name_conflict",
            event_info=event_info,
            conflicting_names=conflicting_names,
        )
