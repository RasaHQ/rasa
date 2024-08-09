from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class LLMResponse:
    data: List[str]
    usage: Dict[str, Any]
    metadata: Dict[str, Any]
