DEFAULT_SELECTIVE_DOMAIN = False
SELECTIVE_DOMAIN = "enable_selective_domain"

SSL_CLIENT_CERT_FIELD = "ssl_client_cert"
SSL_CLIENT_KEY_FIELD = "ssl_client_key"

# Special marker key used by EndpointConfig to indicate 449 status
# without raising an exception
MISSING_DOMAIN_MARKER = "missing_domain"

SESSION_START_REJECTION_MESSAGE = (
    "Session was already started for the current message. "
    "Skipping execution of action_session_start."
)

STREAMING_QUEUE_MAX_SIZE = 32

# Prefix of the StreamError.message the rasa-sdk sends when it requires the
# domain to be included in the streaming request.  Used by
# GRPCCustomActionExecutor.run_streaming() to raise DomainNotFound instead of
# a generic RasaException so that RetryCustomActionExecutor can retry with the
# domain payload attached.
STREAM_ERROR_MISSING_DOMAIN = "Missing domain context"
