from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from rasa.shared.core.events import Event


class AgentInputSlot(BaseModel):
    """A class that represents the schema of the input slot to the agent."""

    name: str
    value: Any
    type: str
    allowed_values: Optional[List[Any]] = None


class AgentInput(BaseModel):
    """A class that represents the schema of the input to the agent."""

    id: str
    user_message: str
    slots: List[AgentInputSlot]
    conversation_history: str
    events: List[Event]
    metadata: Dict[str, Any]
    timestamp: Optional[str] = None
    recipient_id: Optional[str] = None

    class Config:
        """Skip validation for Event class as pydantic does not know how to
        serialize or handle instances of the class.
        """

        arbitrary_types_allowed = True

    @property
    def slot_names(self) -> List[str]:
        """Get the names of the slots in the input."""
        return [slot.name for slot in self.slots]
