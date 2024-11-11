from rasa.shared.providers._configs.self_hosted_llm_client_config import SelfHostedLLMClientConfig

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import structlog

@dataclass
class RasaLLMClientConfig(SelfHostedLLMClientConfig):
    """Parses configuration for a Rasa Hosted LiteLLM client, resolves aliases and
    raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
    """