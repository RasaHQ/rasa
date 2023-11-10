from dataclasses import dataclass

from rasa.dialogue_understanding.stack.frames import BaseFlowStackFrame


@dataclass
class PatternFlowStackFrame(BaseFlowStackFrame):
    """A stack frame that represents a pattern flow."""

    pass
