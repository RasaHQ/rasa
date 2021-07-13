DEFAULT_SERVER_PORT = 5005

DEFAULT_SERVER_FORMAT = "{}://localhost:{}"

DEFAULT_SERVER_URL = DEFAULT_SERVER_FORMAT.format("http", DEFAULT_SERVER_PORT)

DEFAULT_NLU_FALLBACK_THRESHOLD = 0.3

DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD = 0.1

DEFAULT_CORE_FALLBACK_THRESHOLD = 0.3

DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes

DEFAULT_RESPONSE_TIMEOUT = 60 * 60  # 1 hour

DEFAULT_LOCK_LIFETIME = 60  # in seconds

BEARER_TOKEN_PREFIX = "Bearer "

# The lowest priority is intended to be used by machine learning policies.
DEFAULT_POLICY_PRIORITY = 1
# The priority intended to be used by memoization policies.
# It is higher than default to prioritize training stories.
MEMOIZATION_POLICY_PRIORITY = 2
# The priority of the `RulePolicy` is higher than all other policies since
# rule execution takes precedence over training stories or predicted actions.
RULE_POLICY_PRIORITY = 3

DIALOGUE = "dialogue"

# RabbitMQ message property header added to events published using `rasa export`
RASA_EXPORT_PROCESS_ID_HEADER_NAME = "rasa-export-process-id"

# Name of the environment variable defining the PostgreSQL schema to access. See
# https://www.postgresql.org/docs/9.1/ddl-schemas.html for more details.
POSTGRESQL_SCHEMA = "POSTGRESQL_SCHEMA"

# Names of the environment variables defining PostgreSQL pool size and max overflow
POSTGRESQL_POOL_SIZE = "SQL_POOL_SIZE"
POSTGRESQL_MAX_OVERFLOW = "SQL_MAX_OVERFLOW"

# File names for testing
CONFUSION_MATRIX_STORIES_FILE = "story_confusion_matrix.png"
REPORT_STORIES_FILE = "story_report.json"
FAILED_STORIES_FILE = "failed_test_stories.yml"
SUCCESSFUL_STORIES_FILE = "successful_test_stories.yml"
STORIES_WITH_WARNINGS_FILE = "stories_with_warnings.yml"
