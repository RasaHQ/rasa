from dataclasses import dataclass
from typing import Any, Dict, List


def command_from_json(data: Dict[str, Any]) -> "Command":
    """Converts a dictionary to a command object.

    Args:
        data: The dictionary to convert.

    Returns:
    The converted command object.
    """
    if data.get("command") == "set slot":
        return SetSlotCommand(name=data["name"], value=data["value"])
    elif data.get("command") == "correct slot":
        return CorrectSlotsCommand(
            corrected_slots=[
                CorrectedSlot(s["name"], value=s["value"])
                for s in data["corrected_slots"]
            ]
        )
    elif data.get("command") == "start flow":
        return StartFlowCommand(flow=data["flow"])
    elif data.get("command") == "cancel flow":
        return CancelFlowCommand()
    elif data.get("command") == "clarify flow":
        return ClarifyNextStepCommand(potential_flows=data["potential_flows"])
    elif data.get("command") == "cant handle":
        return CantHandleCommand()
    elif data.get("command") == "interruption":
        return HandleInterruptionCommand()
    elif data.get("command") == "listen":
        return ListenCommand()
    elif data.get("command") == "human handoff":
        return HumanHandoffCommand()
    else:
        raise ValueError(f"Unknown command type: {data}")


@dataclass
class Command:
    """A command that can be executed on a tracker."""

    @property
    def command(self) -> str:
        """Returns the command type."""
        raise NotImplementedError()


@dataclass
class SetSlotCommand(Command):
    """A command to set a slot."""

    name: str
    value: Any
    command: str = "set slot"


@dataclass
class CorrectedSlot:
    """A slot that was corrected."""

    name: str
    value: Any


@dataclass
class CorrectSlotsCommand(Command):
    """A command to correct the value of a slot."""

    corrected_slots: List[CorrectedSlot]

    command: str = "correct slot"


@dataclass
class StartFlowCommand(Command):
    """A command to start a flow."""

    flow: str
    command: str = "start flow"


@dataclass
class CancelFlowCommand(Command):
    """A command to cancel the current flow."""

    command: str = "cancel flow"


@dataclass
class ClarifyNextStepCommand(Command):
    """A command to clarify the flow to be run."""

    potential_flows: List[str]
    command: str = "clarify flow"


@dataclass
class CantHandleCommand(Command):
    """A command to indicate that the bot can't handle the user's input."""

    command: str = "cant handle"


@dataclass
class HandleInterruptionCommand(Command):
    """A command to indicate that the bot was interrupted."""

    command: str = "interruption"


@dataclass
class ListenCommand(Command):
    """A command to indicate that the bot should not respond but listen."""

    command: str = "listen"


@dataclass
class HumanHandoffCommand(Command):
    """A command to indicate that the bot should handoff to a human."""

    command: str = "human handoff"


@dataclass
class ErrorCommand(Command):
    """A command to indicate that the bot failed to handle the dialogue."""

    command: str = "error"
