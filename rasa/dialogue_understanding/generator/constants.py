from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED,
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
)

DEFAULT_LLM_CONFIG = {
    "_type": "openai",
    "request_timeout": 7,
    "temperature": 0.0,
    "model_name": DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED,
    "max_tokens": DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
}

LLM_CONFIG_KEY = "llm"
USER_INPUT_CONFIG_KEY = "user_input"

FLOW_RETRIEVAL_KEY = "flow_retrieval"
FLOW_RETRIEVAL_ACTIVE_KEY = "active"
