from typing import Literal

# A dot-path for importlib to the copilot prompt
COPILOT_PROMPTS_DIR = "builder.copilot.prompts"
COPILOT_PROMPTS_FILE = "copilot_system_prompt.jinja2"
COPILOT_LAST_USER_MESSAGE_CONTEXT_PROMPT_FILE = (
    "latest_user_message_context_prompt.jinja2"
)
COPILOT_TRAINING_ERROR_HANDLER_PROMPT_FILE = (
    "copilot_training_error_handler_prompt.jinja2"
)

# A dot-path for importlib to the rasa internal messages templates
COPILOT_MESSAGE_TEMPLATES_DIR = "builder.copilot.templated_messages"
RASA_INTERNAL_MESSAGES_TEMPLATES_FILE = "copilot_internal_messages_templates.yml"
COPILOT_HANDLER_RESPONSES_FILE = "copilot_templated_responses.yml"
COPILOT_WELCOME_MESSAGES_FILE = "copilot_welcome_messages.yml"
COPILOT_TEMPLATE_PROMPTS_FILE = "copilot_template_prompts.yml"

# OpenAI roles copilot utilizes - Use literal types to avoid type errors with OpenAI
ROLE_USER: Literal["user"] = "user"
ROLE_SYSTEM: Literal["system"] = "system"
ROLE_ASSISTANT: Literal["assistant"] = "assistant"

# Rasa Copilot role - Added to avoid confusion with the assistant role on the frontend.
ROLE_COPILOT: Literal["copilot"] = "copilot"

# Rasa internal role - Used to indicate that the message is from the Rasa internal
# system components.
ROLE_COPILOT_INTERNAL: Literal["internal_copilot_request"] = "internal_copilot_request"

# Copilot Telemetry
COPILOT_SEGMENT_WRITE_KEY_ENV_VAR = "COPILOT_SEGMENT_WRITE_KEY"

# Chat ID used to store default copilot history when no specific chat ID is provided.
DEFAULT_COPILOT_CHAT_ID = "default"

# Welcome message key for prompt-to-bot (not a template enum)
PROMPT_TO_BOT_KEY = "prompt_to_bot"
