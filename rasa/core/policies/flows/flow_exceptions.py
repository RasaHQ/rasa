from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.exceptions import RasaException


class FlowException(RasaException):
    """Exception that is raised when there is a problem with a flow."""

    def __init__(self, message: str = "Flow error occurred") -> None:
        """Initialize FlowException with a message."""
        super().__init__(message)


class FlowCircuitBreakerTrippedException(FlowException):
    """Exception that is raised when the flow circuit breaker tripped.

    The circuit breaker gets tripped when a flow seems to be stuck in
    executing steps and does not make any progress.
    """

    def __init__(
        self, dialogue_stack: DialogueStack, number_of_steps_taken: int
    ) -> None:
        """Creates a `FlowCircuitBreakerTrippedException`.

        Args:
            dialogue_stack: The dialogue stack.
            number_of_steps_taken: The number of steps that were taken.
        """
        super().__init__(
            f"Flow circuit breaker tripped after {number_of_steps_taken} steps. "
            "There appears to be an infinite loop in the flows."
        )
        self.dialogue_stack = dialogue_stack
        self.number_of_steps_taken = number_of_steps_taken


class NoNextStepInFlowException(FlowException):
    """Exception that is raised when there is no next step in a flow."""

    def __init__(self, dialogue_stack: DialogueStack) -> None:
        """Creates a `NoNextStepInFlowException`.

        Args:
            dialogue_stack: The dialogue stack.
        """
        super().__init__("No next step can be selected for the flow.")
        self.dialogue_stack = dialogue_stack
