DEFAULT_SERVER_PORT = 5005

DEFAULT_SERVER_FORMAT = "{}://localhost:{}"

DEFAULT_SERVER_URL = DEFAULT_SERVER_FORMAT.format("http", DEFAULT_SERVER_PORT)

DEFAULT_NLU_FALLBACK_THRESHOLD = 0.0

DEFAULT_CORE_FALLBACK_THRESHOLD = 0.0

DEFAULT_FALLBACK_ACTION = "action_default_fallback"

DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes

DEFAULT_LOCK_LIFETIME = 60  # in seconds

REQUESTED_SLOT = "requested_slot"

# slots for knowledge base
SLOT_LISTED_ITEMS = "knowledge_base_listed_objects"
SLOT_LAST_OBJECT = "knowledge_base_last_object"
SLOT_LAST_OBJECT_TYPE = "knowledge_base_last_object_type"
DEFAULT_KNOWLEDGE_BASE_ACTION = "action_query_knowledge_base"

# start of special user message section
INTENT_MESSAGE_PREFIX = "/"
EXTERNAL_MESSAGE_PREFIX = "EXTERNAL: "

USER_INTENT_RESTART = "restart"

USER_INTENT_SESSION_START = "session_start"

USER_INTENT_BACK = "back"

USER_INTENT_OUT_OF_SCOPE = "out_of_scope"

ACTION_NAME_SENDER_ID_CONNECTOR_STR = "__sender_id:"

BEARER_TOKEN_PREFIX = "Bearer "

# Key to access data in the event metadata which specifies if an event was caused by an external entity (e.g. a sensor).
IS_EXTERNAL = "is_external"

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
UTTER_PREFIX = "utter_"
RESPOND_PREFIX = "respond_"
