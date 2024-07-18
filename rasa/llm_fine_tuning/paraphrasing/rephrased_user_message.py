from dataclasses import dataclass, field
from typing import List


@dataclass
class RephrasedUserMessage:
    original_user_message: str
    rephrasings: List[str]
    failed_rephrasings: List[str] = field(default_factory=list)
    passed_rephrasings: List[str] = field(default_factory=list)
