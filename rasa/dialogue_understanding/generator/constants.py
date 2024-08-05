from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED,
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
)

DEFAULT_LLM_CONFIG = {
    "api_type": "openai",
    "model": DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED,
    "temperature": 0.0,
    "max_tokens": DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    "request_timeout": 7,
}

LLM_CONFIG_KEY = "llm"
USER_INPUT_CONFIG_KEY = "user_input"

FLOW_RETRIEVAL_KEY = "flow_retrieval"
FLOW_RETRIEVAL_ACTIVE_KEY = "active"
