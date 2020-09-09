DEFAULT_SERVER_PORT = 5005

DEFAULT_SERVER_FORMAT = "{}://localhost:{}"

DEFAULT_SERVER_URL = DEFAULT_SERVER_FORMAT.format("http", DEFAULT_SERVER_PORT)

DEFAULT_NLU_FALLBACK_THRESHOLD = 0.0

DEFAULT_CORE_FALLBACK_THRESHOLD = 0.0

DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes

DEFAULT_RESPONSE_TIMEOUT = 60 * 60  # 1 hour

DEFAULT_LOCK_LIFETIME = 60  # in seconds

# rules allow setting a value of slots or active_loops to None;
# generator substitutes `None`s with this constant to notify rule policy that
# a value should not be set during prediction to activate a rule

# slots for knowledge base

# start of special user message section

BEARER_TOKEN_PREFIX = "Bearer "

# Key to access data in the event metadata
# It specifies if an event was caused by an external entity (e.g. a sensor).

# the lowest priority intended to be used by machine learning policies
DEFAULT_POLICY_PRIORITY = 1
# the priority intended to be used by mapping policies
MAPPING_POLICY_PRIORITY = 2
# the priority intended to be used by memoization policies
# it is higher than default and mapping to prioritize training stories
MEMOIZATION_POLICY_PRIORITY = 3
# the priority intended to be used by fallback policies
# it is higher than memoization to prioritize fallback
FALLBACK_POLICY_PRIORITY = 4
# the priority intended to be used by form policies
# it is the highest to prioritize form to the rest of the policies
FORM_POLICY_PRIORITY = 5
RESPOND_PREFIX = "respond_"

DIALOGUE = "dialogue"

# RabbitMQ message property header added to events published using `rasa export`
RASA_EXPORT_PROCESS_ID_HEADER_NAME = "rasa-export-process-id"

# Name of the environment variable defining the PostgreSQL schema to access. See
# https://www.postgresql.org/docs/9.1/ddl-schemas.html for more details.
POSTGRESQL_SCHEMA = "POSTGRESQL_SCHEMA"

# Names of the environment variables defining PostgreSQL pool size and max overflow
POSTGRESQL_POOL_SIZE = "SQL_POOL_SIZE"
POSTGRESQL_MAX_OVERFLOW = "SQL_MAX_OVERFLOW"

# the keys for State (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# represent the origin of a SubState
USER = "user"
SLOTS = "slots"
