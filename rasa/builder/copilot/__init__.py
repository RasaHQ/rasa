from typing import Type

from rasa.builder import config
from rasa.builder.copilot.base_copilot import BaseCopilot
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler

if config.USE_AGENT_SDK_COPILOT:
    from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

    Copilot: Type[BaseCopilot] = AgentCopilot
else:
    from rasa.builder.copilot.legacy_copilot import LegacyCopilot

    Copilot: Type[BaseCopilot] = LegacyCopilot  # type: ignore[no-redef]

__all__ = [
    "Copilot",
    "CopilotResponseHandler",
    "BaseCopilot",
]
