from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from rasa.agents.core.types import AgentStatus
from rasa.shared.core.events import Event


class AgentOutput(BaseModel):
    """A class that represents the schema of the output from the agent."""

    id: str
    status: AgentStatus
    response_message: Optional[str] = None
    events: Optional[List[Event]] = None
    structured_results: Optional[List[List[Dict[str, Any]]]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        """Skip validation for SlotSet class as pydantic does not know how to
        serialize or handle instances of the class.
        """

        arbitrary_types_allowed = True
