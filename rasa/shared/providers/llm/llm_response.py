from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class LLMUsage:
    prompt_tokens: int
    """Number of prompt tokens used to generate completion."""

    completion_tokens: int
    """Number of generated tokens."""

    total_tokens: int = field(init=False)
    """Total number of used tokens."""

    def __post_init__(self) -> None:
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        """Converts the LLMUsage dataclass instance into a dictionary."""
        return asdict(self)


@dataclass
class LLMResponse:
    id: str
    """A unique identifier for the completion."""

    choices: List[str]
    """The list of completion choices the model generated for the input prompt."""

    created: int
    """The Unix timestamp (in seconds) of when the completion was created."""

    model: Optional[str] = None
    """The model used for completion."""

    usage: Optional[LLMUsage] = None
    """An optional details about the token usage for the API call."""

    additional_info: Optional[Dict] = None
    """Optional dictionary for storing additional information related to the
    completion that may not be covered by other fields."""

    def to_dict(self) -> dict:
        """Converts the LLMResponse dataclass instance into a dictionary."""
        result = asdict(self)
        if self.usage:
            result["usage"] = self.usage.to_dict()
        return result
