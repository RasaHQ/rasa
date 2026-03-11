"""Context type passed to custom tool executors."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class AgentToolContext(BaseModel):
    """Context passed to custom tool executors.

    Attributes:
        metadata: Request metadata (e.g. from AgentInput).
    """

    metadata: Dict[str, Any] = Field(default_factory=dict)
