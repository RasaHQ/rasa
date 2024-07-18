from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class EmbeddingUsage:
    prompt_tokens: int
    """Number of prompt tokens used to generate completion."""

    completion_tokens: int
    """Number of generated tokens."""

    total_tokens: int
    """Total number of used tokens."""

    def to_dict(self) -> dict:
        """Converts the EmbeddingUsage dataclass instance into a dictionary."""
        return asdict(self)


@dataclass
class EmbeddingResponse:
    data: List[List[float]]
    """The embedding data returned by the API call."""

    model: Optional[str] = None
    """The model used for the embedding."""

    usage: Optional[EmbeddingUsage] = None
    """An optional details about the token usage for the API call."""

    additional_info: Optional[Dict] = None
    """Optional dictionary for storing additional information related to the
    completion that may not be covered by other fields."""

    def to_dict(self) -> dict:
        """Converts the EmbeddingResponse dataclass instance into a dictionary."""
        result = asdict(self)
        if self.usage:
            result["usage"] = self.usage.to_dict()
        return result
