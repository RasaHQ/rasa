from typing import Type

from rasa.builder import config
from rasa.builder.copilot.base_copilot import BaseCopilot
from rasa.builder.copilot.response_handling.base_copilot_response_handler import (
    BaseCopilotResponseHandler,
)

if config.USE_AGENT_SDK_COPILOT:
    from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot
    from rasa.builder.copilot.response_handling.agent_copilot_response_handler import (
        AgentCopilotResponseHandler,
    )

    Copilot: Type[BaseCopilot] = AgentCopilot
    CopilotResponseHandler: Type[BaseCopilotResponseHandler] = (
        AgentCopilotResponseHandler
    )
else:
    from rasa.builder.copilot.legacy_copilot import LegacyCopilot
    from rasa.builder.copilot.response_handling.legacy_copilot_response_handler import (
        LegacyCopilotResponseHandler,
    )

    Copilot: Type[BaseCopilot] = LegacyCopilot  # type: ignore[no-redef]
    CopilotResponseHandler: Type[BaseCopilotResponseHandler] = (  # type: ignore[no-redef]
        LegacyCopilotResponseHandler
    )


__all__ = [
    "Copilot",
    "CopilotResponseHandler",
    "BaseCopilot",
    "BaseCopilotResponseHandler",
]
