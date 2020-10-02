import rasa.shared.constants as constants

DEFAULT_CATEGORICAL_SLOT_VALUE = "__other__"

USER_INTENT_RESTART = "restart"
USER_INTENT_BACK = "back"
USER_INTENT_OUT_OF_SCOPE = "out_of_scope"
USER_INTENT_SESSION_START = "session_start"

DEFAULT_INTENTS = [
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_OUT_OF_SCOPE,
    USER_INTENT_SESSION_START,
    constants.DEFAULT_NLU_FALLBACK_INTENT_NAME,
]

LOOP_NAME = "name"

ACTION_LISTEN_NAME = "action_listen"
ACTION_RESTART_NAME = "action_restart"
ACTION_SESSION_START_NAME = "action_session_start"
ACTION_DEFAULT_FALLBACK_NAME = "action_default_fallback"
ACTION_DEACTIVATE_LOOP_NAME = "action_deactivate_loop"
LEGACY_ACTION_DEACTIVATE_LOOP_NAME = "action_deactivate_form"
ACTION_REVERT_FALLBACK_EVENTS_NAME = "action_revert_fallback_events"
ACTION_DEFAULT_ASK_AFFIRMATION_NAME = "action_default_ask_affirmation"
ACTION_DEFAULT_ASK_REPHRASE_NAME = "action_default_ask_rephrase"
ACTION_BACK_NAME = "action_back"
ACTION_TWO_STAGE_FALLBACK_NAME = "action_two_stage_fallback"
RULE_SNIPPET_ACTION_NAME = "..."

DEFAULT_ACTION_NAMES = [
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_DEACTIVATE_LOOP_NAME,
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
    ACTION_BACK_NAME,
    RULE_SNIPPET_ACTION_NAME,
]

# rules allow setting a value of slots or active_loops to None;
# generator substitutes `None`s with this constant to notify rule policy that
# a value should not be set during prediction to activate a rule
SHOULD_NOT_BE_SET = "should_not_be_set"

PREVIOUS_ACTION = "prev_action"
ACTIVE_LOOP = "active_loop"
LOOP_INTERRUPTED = "is_interrupted"
LOOP_REJECTED = "rejected"
TRIGGER_MESSAGE = "trigger_message"

# start of special user message section
EXTERNAL_MESSAGE_PREFIX = "EXTERNAL: "
# Key to access data in the event metadata
# It specifies if an event was caused by an external entity (e.g. a sensor).
IS_EXTERNAL = "is_external"

ACTION_NAME_SENDER_ID_CONNECTOR_STR = "__sender_id:"

REQUESTED_SLOT = "requested_slot"

# slots for knowledge base
SLOT_LISTED_ITEMS = "knowledge_base_listed_objects"
SLOT_LAST_OBJECT = "knowledge_base_last_object"
SLOT_LAST_OBJECT_TYPE = "knowledge_base_last_object_type"
DEFAULT_KNOWLEDGE_BASE_ACTION = "action_query_knowledge_base"

# the keys for `State` (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# represent the origin of a `SubState`
USER = "user"
SLOTS = "slots"
