from __future__ import annotations

from typing import Optional

from rasa.shared.exceptions import RasaException


class UnreachableFlowStepException(RasaException):
    """Raised when a flow step is unreachable."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step_id}' in flow '{self.flow_id}' can not be reached "
            f"from the start step. Please make sure that all steps can be reached "
            f"from the start step, e.g. by "
            f"checking that another step points to this step."
        )


class MissingNextLinkException(RasaException):
    """Raised when a flow step is missing a next link."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step_id}' in flow '{self.flow_id}' is missing a `next`. "
            f"As a last step of a branch, it is required to have one. "
        )


class ReservedFlowStepIdException(RasaException):
    """Raised when a flow step is using a reserved id."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step_id}' in flow '{self.flow_id}' is using the reserved id "
            f"'{self.step_id}'. Please use a different id for your step."
        )


class MissingElseBranchException(RasaException):
    """Raised when a flow step is missing an else branch."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step_id}' in flow '{self.flow_id}' is missing an `else` "
            f"branch. If a steps `next` statement contains an `if` it always "
            f"also needs an `else` branch. Please add the missing `else` branch."
        )


class NoNextAllowedForLinkException(RasaException):
    """Raised when a flow step has a next link but is not allowed to have one."""

    def __init__(self, step_id: str, flow_id: str) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Link step '{self.step_id}' in flow '{self.flow_id}' has a `next` but "
            f"as a link step is not allowed to have one."
        )


class UnresolvedFlowStepIdException(RasaException):
    """Raised when a flow step is referenced, but its id can not be resolved."""

    def __init__(
        self, step_id: str, flow_id: str, referenced_from_step_id: Optional[str]
    ) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow_id = flow_id
        self.referenced_from_step_id = referenced_from_step_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        if self.referenced_from_step_id:
            exception_message = (
                f"Step with id '{self.step_id}' could not be resolved. "
                f"'Step '{self.referenced_from_step_id}' in flow '{self.flow_id}' "
                f"referenced this step but it does not exist. "
            )
        else:
            exception_message = (
                f"Step '{self.step_id}' in flow '{self.flow_id}' can not be resolved. "
            )

        return exception_message + (
            "Please make sure that the step is defined in the same flow."
        )


class UnresolvedFlowException(RasaException):
    """Raised when a flow is referenced, but its id cannot be resolved."""

    def __init__(self, flow_id: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return (
            f"Flow '{self.flow_id}' can not be resolved. "
            f"Please make sure that the flow is defined."
        )


class EmptyStepSequenceException(RasaException):
    """Raised when an empty step sequence is encountered."""

    def __init__(self, flow_id: str, step_id: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id
        self.step_id = step_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        if not self.flow_id:
            return "Encountered an empty step sequence."
        else:
            return f"Encountered an empty step sequence in flow '{self.flow_id}'."


class EmptyFlowException(RasaException):
    """Raised when a flow is completely empty."""

    def __init__(self, flow_id: str) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return f"Flow '{self.flow_id}' does not have any steps."
