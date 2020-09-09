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
ACTION_DEACTIVATE_FORM_NAME = "action_deactivate_form"
ACTION_REVERT_FALLBACK_EVENTS_NAME = "action_revert_fallback_events"
ACTION_DEFAULT_ASK_AFFIRMATION_NAME = "action_default_ask_affirmation"
ACTION_DEFAULT_ASK_REPHRASE_NAME = "action_default_ask_rephrase"
ACTION_BACK_NAME = "action_back"
ACTION_TWO_STAGE_FALLBACK_NAME = "two_stage_fallback"
RULE_SNIPPET_ACTION_NAME = "..."

DEFAULT_ACTIONS = [
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_DEACTIVATE_FORM_NAME,
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
    ACTION_BACK_NAME,
    RULE_SNIPPET_ACTION_NAME,
]

SHOULD_NOT_BE_SET = "should_not_be_set"
PREVIOUS_ACTION = "prev_action"
ACTIVE_LOOP = "active_loop"
LOOP_VALIDATE = "validate"
LOOP_REJECTED = "rejected"
TRIGGER_MESSAGE = "trigger_message"

EXTERNAL_MESSAGE_PREFIX = "EXTERNAL: "
ACTION_NAME_SENDER_ID_CONNECTOR_STR = "__sender_id:"
IS_EXTERNAL = "is_external"
REQUESTED_SLOT = "requested_slot"
SLOT_LISTED_ITEMS = "knowledge_base_listed_objects"
SLOT_LAST_OBJECT = "knowledge_base_last_object"
SLOT_LAST_OBJECT_TYPE = "knowledge_base_last_object_type"
DEFAULT_KNOWLEDGE_BASE_ACTION = "action_query_knowledge_base"
UTTER_PREFIX = "utter_"

# the keys for State (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# represent the origin of a SubState
USER = "user"
SLOTS = "slots"
