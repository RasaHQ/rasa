DEFAULT_SERVER_PORT = 5005

DEFAULT_SERVER_FORMAT = "http://localhost:{}"

DEFAULT_SERVER_URL = DEFAULT_SERVER_FORMAT.format(DEFAULT_SERVER_PORT)

DEFAULT_NLU_FALLBACK_THRESHOLD = 0.0

DEFAULT_CORE_FALLBACK_THRESHOLD = 0.0

DEFAULT_FALLBACK_ACTION = "action_default_fallback"

DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes

REQUESTED_SLOT = "requested_slot"

# slots for knowledge base
SLOT_LISTED_ITEMS = "knowledge_base_listed_items"
SLOT_LAST_ENTITY = "knowledge_base_last_entity"
SLOT_LAST_ENTITY_TYPE = "knowledge_base_last_entity_type"
DEFAULT_KNOWLEDGE_BASE_ACTION = "action_query_knowledge_base"

# start of special user message section
INTENT_MESSAGE_PREFIX = "/"

USER_INTENT_RESTART = "restart"

USER_INTENT_BACK = "back"

USER_INTENT_OUT_OF_SCOPE = "out_of_scope"

ACTION_NAME_SENDER_ID_CONNECTOR_STR = "__sender_id:"

BEARER_TOKEN_PREFIX = "Bearer "
