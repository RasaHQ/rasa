# Agent configuration constants
AGENT_DEFAULT_MAX_RETRIES = 5
AGENT_DEFAULT_TIMEOUT_SECONDS = 30
MAX_AGENT_RETRY_DELAY_SECONDS = 5

# MCP Tool related constants (primary)
TOOL_TYPE_KEY = "type"
TOOL_NAME_KEY = "name"
TOOL_DESCRIPTION_KEY = "description"
TOOL_PARAMETERS_KEY = "parameters"
TOOL_STRICT_KEY = "strict"
TOOL_TYPE_FUNCTION_KEY = "function"
TOOL_EXECUTOR_KEY = "tool_executor"

# MCP Tool related constants (secondary)
TOOL_ADDITIONAL_PROPERTIES_KEY = "additionalProperties"
TOOL_PROPERTIES_KEY = "properties"
TOOL_REQUIRED_KEY = "required"
TOOL_PROPERTY_TYPE_KEY = "type"

# MCP Tool Anthropic format related constants
TOOL_INPUT_SCHEMA_KEY = "input_schema"

# MCP Message related constants
KEY_ROLE = "role"
KEY_CONTENT = "content"

# MCP Tool call related constants
KEY_TOOL_CALL_ID = "tool_call_id"
KEY_FUNCTION = "function"
KEY_NAME = "name"
KEY_ARGUMENTS = "arguments"
KEY_ID = "id"
KEY_TYPE = "type"
KEY_TOOL_CALLS = "tool_calls"

# Agent output metadata related constants
AGENT_METADATA_AGENT_RESPONSE_KEY = "agent_response"
AGENT_METADATA_STRUCTURED_RESULTS_KEY = "structured_results"
AGENT_METADATA_EXIT_IF_KEY = "exit_if"
AGENT_METADATA_SENDER_ID_KEY = "sender_id"
AGENT_METADATA_AGENT_ID_KEY = "agent_id"
AGENT_METADATA_MODEL_ID_KEY = "model_id"
# A2A-specific constants
A2A_AGENT_CONTEXT_ID_KEY = "context_id"
A2A_AGENT_TASK_ID_KEY = "task_id"
