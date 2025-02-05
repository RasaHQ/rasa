import functools
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Text, Union

import structlog

structlogger = structlog.get_logger()


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

    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> "LLMUsage":
        """Creates an LLMUsage object from a dictionary.
        If any keys are missing, they will default to zero
        or whatever default you prefer.
        """
        return cls(
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
        )

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

    latency: Optional[float] = None
    """Optional field to store the latency of the LLM API call."""

    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> "LLMResponse":
        """Creates an LLMResponse from a dictionary."""
        usage_data = data.get("usage", {})
        usage_obj = LLMUsage.from_dict(usage_data) if usage_data else None

        return cls(
            id=data["id"],
            choices=data["choices"],
            created=data["created"],
            model=data.get("model"),
            usage=usage_obj,
            additional_info=data.get("additional_info"),
            latency=data.get("latency"),
        )

    @classmethod
    def ensure_llm_response(cls, response: Union[str, "LLMResponse"]) -> "LLMResponse":
        if isinstance(response, LLMResponse):
            return response

        structlogger.warn("llm_response.deprecated_response_type", response=response)
        data = {"id": None, "choices": [response], "created": None}
        return LLMResponse.from_dict(data)

    def to_dict(self) -> dict:
        """Converts the LLMResponse dataclass instance into a dictionary."""
        result = asdict(self)
        if self.usage:
            result["usage"] = self.usage.to_dict()
        return result


def measure_llm_latency(
    func: Callable[..., Awaitable[Optional[LLMResponse]]],
) -> Callable[..., Awaitable[Optional[LLMResponse]]]:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Optional[LLMResponse]:
        start = time.perf_counter()
        result: Optional[LLMResponse] = await func(*args, **kwargs)
        if result:
            result.latency = time.perf_counter() - start
        return result

    return wrapper
