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
    elif data.get("command") == "cant handle":
        return CantHandleCommand()
    elif data.get("command") == "chitchat":
        return ChitChatAnswerCommand()
    elif data.get("command") == "knowledge":
        return KnowledgeAnswerCommand()
    elif data.get("command") == "human handoff":
        return HumanHandoffCommand()
    elif data.get("command") == "clarify":
        return ClarifyCommand(options=data["options"])
    elif data.get("command") == "error":
        return ErrorCommand()
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
class CantHandleCommand(Command):
    """A command to indicate that the bot can't handle the user's input."""

    command: str = "cant handle"


@dataclass
class FreeFormAnswerCommand(Command):
    """A command to indicate a free-form answer by the bot."""

    command: str = "free form answer"


@dataclass
class ChitChatAnswerCommand(FreeFormAnswerCommand):
    """A command to indicate a chitchat style free-form answer by the bot."""

    command: str = "chitchat"


@dataclass
class KnowledgeAnswerCommand(FreeFormAnswerCommand):
    """A command to indicate a knowledge-based free-form answer by the bot."""

    command: str = "knowledge"


@dataclass
class HumanHandoffCommand(Command):
    """A command to indicate that the bot should handoff to a human."""

    command: str = "human handoff"


@dataclass
class ErrorCommand(Command):
    """A command to indicate that the bot failed to handle the dialogue."""

    command: str = "error"


@dataclass
class ClarifyCommand(Command):
    """A command to indicate that the bot should ask for clarification."""

    options: List[str]
    command: str = "clarify"
