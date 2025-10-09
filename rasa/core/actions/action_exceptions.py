from typing import Optional

from rasa.shared.exceptions import RasaException


class ActionExecutionRejection(RasaException):
    """Raising this exception allows other policies to predict a different action."""

    def __init__(self, action_name: str, message: Optional[str] = None) -> None:
        """Create a new ActionExecutionRejection exception."""
        self.action_name = action_name
        self.message = message or "Custom action '{}' rejected to run".format(
            action_name
        )
        super(ActionExecutionRejection, self).__init__(self.message)

    def __str__(self) -> str:
        return self.message


class DomainNotFound(Exception):
    """Exception raised when domain is not found."""

    pass
