# endpoints keys
ENDPOINTS_TRACING_KEY = "tracing"
ENDPOINTS_TRACING_SERVICE_NAME_KEY = "service_name"
ENDPOINTS_METRICS_KEY = "metrics"
ENDPOINTS_ROOT_CERTIFICATES_KEY = "root_certificates"
ENDPOINTS_INSECURE_KEY = "insecure"
ENDPOINTS_ENDPOINT_KEY = "endpoint"
ENDPOINTS_OTLP_BACKEND_TYPE = "otlp"

# tracing attributes
PROMPT_TOKEN_LENGTH_ATTRIBUTE_NAME = "len_prompt_tokens"
REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME = "request_body_size_in_bytes"

# Tool output value truncation
# Truncate slot values to prevent excessive memory usage in traces
# while preserving enough context for debugging
TOOL_OUTPUT_VALUE_MAX_LENGTH = 500

# metrics constants
LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME = "llm_command_generator_cpu_usage"
LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME = "llm_command_generator_memory_usage"
LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME = (
    "llm_command_generator_prompt_token_usage"
)
LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME = (
    "llm_command_generator_llm_response_duration"
)
SINGLE_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME = (
    "single_step_llm_command_generator_cpu_usage"
)
SINGLE_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME = (
    "single_step_llm_command_generator_memory_usage"
)
SINGLE_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME = (
    "single_step_llm_command_generator_prompt_token_usage"
)
SINGLE_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME = (
    "single_step_llm_command_generator_llm_response_duration"
)
COMPACT_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME = (
    "compact_llm_command_generator_cpu_usage"
)
COMPACT_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME = (
    "compact_llm_command_generator_memory_usage"
)
COMPACT_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME = (
    "compact_llm_command_generator_prompt_token_usage"
)
COMPACT_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME = (
    "compact_llm_command_generator_llm_response_duration"
)
SEARCH_READY_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME = (
    "search_ready_llm_command_generator_cpu_usage"
)
SEARCH_READY_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME = (
    "search_ready_llm_command_generator_memory_usage"
)
SEARCH_READY_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME = (
    "search_ready_llm_command_generator_prompt_token_usage"
)
SEARCH_READY_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME = (
    "search_ready_llm_command_generator_llm_response_duration"
)
MULTI_STEP_LLM_COMMAND_GENERATOR_CPU_USAGE_METRIC_NAME = (
    "multi_step_llm_command_generator_cpu_usage"
)
MULTI_STEP_LLM_COMMAND_GENERATOR_MEMORY_USAGE_METRIC_NAME = (
    "multi_step_llm_command_generator_memory_usage"
)
MULTI_STEP_LLM_COMMAND_GENERATOR_PROMPT_TOKEN_USAGE_METRIC_NAME = (
    "multi_step_llm_command_generator_prompt_token_usage"
)
MULTI_STEP_LLM_COMMAND_GENERATOR_LLM_RESPONSE_DURATION_METRIC_NAME = (
    "multi_step_llm_command_generator_llm_response_duration"
)
ENTERPRISE_SEARCH_POLICY_CPU_USAGE_METRIC_NAME = "enterprise_search_policy_cpu_usage"
ENTERPRISE_SEARCH_POLICY_MEMORY_USAGE_METRIC_NAME = (
    "enterprise_search_policy_memory_usage"
)
ENTERPRISE_SEARCH_POLICY_PROMPT_TOKEN_USAGE_METRIC_NAME = (
    "enterprise_search_policy_prompt_token_usage"
)
ENTERPRISE_SEARCH_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME = (
    "enterprise_search_policy_llm_response_duration"
)
INTENTLESS_POLICY_LLM_RESPONSE_DURATION_METRIC_NAME = (
    "intentless_policy_llm_response_duration"
)
CONTEXTUAL_RESPONSE_REPHRASER_LLM_RESPONSE_DURATION_METRIC_NAME = (
    "contextual_nlg_llm_response_duration"
)

RASA_CLIENT_REQUEST_DURATION_METRIC_NAME = "rasa_client_request_duration"
RASA_CLIENT_REQUEST_BODY_SIZE_METRIC_NAME = "rasa_client_request_body_size"

# Agent and MCP tool execution duration metrics
AGENT_EXECUTION_DURATION_METRIC_NAME = "agent_execution_duration"
MCP_TOOL_EXECUTION_DURATION_METRIC_NAME = "mcp_tool_execution_duration"

# MCP Agent LLM metrics
MCP_AGENT_LLM_CPU_USAGE_METRIC_NAME = "mcp_agent_llm_cpu_usage"
MCP_AGENT_LLM_MEMORY_USAGE_METRIC_NAME = "mcp_agent_llm_memory_usage"
MCP_AGENT_LLM_PROMPT_TOKEN_USAGE_METRIC_NAME = "mcp_agent_llm_prompt_token_usage"
MCP_AGENT_LLM_RESPONSE_DURATION_METRIC_NAME = "mcp_agent_llm_response_duration"
LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME = "percentage"
DURATION_UNIT_NAME = "ms"

# langfuse constants
LANGFUSE_CONFIG_PUBLIC_KEY_KEY = "public_key"
LANGFUSE_CONFIG_PRIVATE_KEY_KEY = "private_key"
LANGFUSE_CONFIG_BASE_URL_KEY = "host"
LANGFUSE_CONFIG_TIMEOUT_KEY = "timeout"
LANGFUSE_CONFIG_DEBUG_KEY = "debug"
LANGFUSE_CONFIG_ENVIRONMENT_KEY = "environment"
LANGFUSE_CONFIG_RELEASE_KEY = "release"
LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY = "media_upload_thread_count"
LANGFUSE_CONFIG_SAMPLE_RATE_KEY = "sample_rate"

# tracing types
TRACING_TYPE_LANGFUSE = "langfuse"
