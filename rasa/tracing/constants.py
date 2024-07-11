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

LLM_BASED_COMMAND_GENERATOR_CPU_MEMORY_USAGE_UNIT_NAME = "percentage"
DURATION_UNIT_NAME = "ms"
