from __future__ import annotations

from enum import Enum

import rasa.shared.constants as constants

DEFAULT_CATEGORICAL_SLOT_VALUE = "__other__"

USER_INTENT_RESTART = "restart"
USER_INTENT_BACK = "back"
USER_INTENT_OUT_OF_SCOPE = "out_of_scope"
USER_INTENT_SESSION_START = "session_start"
USER_INTENT_SESSION_END = "session_end"
USER_INTENT_SILENCE_TIMEOUT = "silence_timeout"
SESSION_START_METADATA_SLOT = "session_started_metadata"
LANGUAGE_SLOT = "language"
MOCKED_DATETIME_SLOT = "mocked_datetime"

DEFAULT_INTENTS = [
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_OUT_OF_SCOPE,
    USER_INTENT_SESSION_START,
    USER_INTENT_SESSION_END,
    USER_INTENT_SILENCE_TIMEOUT,
    constants.DEFAULT_NLU_FALLBACK_INTENT_NAME,
]

LOOP_NAME = "name"

ACTION_LISTEN_NAME = "action_listen"
ACTION_RESTART_NAME = "action_restart"
ACTION_SEND_TEXT_NAME = "action_send_text"
ACTION_AGENT_REQUEST_USER_INPUT_NAME = "action_agent_request_user_input"
ACTION_SESSION_START_NAME = "action_session_start"
ACTION_DEFAULT_FALLBACK_NAME = "action_default_fallback"
ACTION_DEACTIVATE_LOOP_NAME = "action_deactivate_loop"
ACTION_REVERT_FALLBACK_EVENTS_NAME = "action_revert_fallback_events"
ACTION_DEFAULT_ASK_AFFIRMATION_NAME = "action_default_ask_affirmation"
ACTION_DEFAULT_ASK_REPHRASE_NAME = "action_default_ask_rephrase"
ACTION_BACK_NAME = "action_back"
ACTION_TWO_STAGE_FALLBACK_NAME = "action_two_stage_fallback"
ACTION_UNLIKELY_INTENT_NAME = "action_unlikely_intent"
RULE_SNIPPET_ACTION_NAME = "..."
ACTION_EXTRACT_SLOTS = "action_extract_slots"
ACTION_VALIDATE_SLOT_MAPPINGS = "action_validate_slot_mappings"
ACTION_CANCEL_FLOW = "action_cancel_flow"
ACTION_CLARIFY_FLOWS = "action_clarify_flows"
ACTION_CORRECT_FLOW_SLOT = "action_correct_flow_slot"
ACTION_RUN_SLOT_REJECTIONS_NAME = "action_run_slot_rejections"
ACTION_CLEAN_STACK = "action_clean_stack"
ACTION_TRIGGER_SEARCH = "action_trigger_search"
ACTION_TRIGGER_CHITCHAT = "action_trigger_chitchat"
ACTION_RESET_ROUTING = "action_reset_routing"
ACTION_HANGUP = "action_hangup"
ACTION_REPEAT_BOT_MESSAGES = "action_repeat_bot_messages"

# pattern continue interrupted flows
ACTION_CONTINUE_INTERRUPTED_FLOW = "action_continue_interrupted_flow"
ACTION_CANCEL_INTERRUPTED_FLOWS = "action_cancel_interrupted_flows"


ACTION_METADATA_EXECUTION_SUCCESS = "execution_success"
ACTION_METADATA_EXECUTION_ERROR_MESSAGE = "execution_error_message"
ACTION_METADATA_EXECUTION_TIME = "execution_times"

ACTION_METADATA_MESSAGE_KEY = "message"
ACTION_METADATA_TEXT_KEY = "text"
ACTION_METADATA_LLM_CONFIG_KEY = "llm_config"
ACTION_METADATA_PROMPT_KEY = "prompt"

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
    ACTION_UNLIKELY_INTENT_NAME,
    ACTION_BACK_NAME,
    ACTION_SEND_TEXT_NAME,
    ACTION_AGENT_REQUEST_USER_INPUT_NAME,
    RULE_SNIPPET_ACTION_NAME,
    ACTION_EXTRACT_SLOTS,
    ACTION_CANCEL_FLOW,
    ACTION_CORRECT_FLOW_SLOT,
    ACTION_CLARIFY_FLOWS,
    ACTION_RUN_SLOT_REJECTIONS_NAME,
    ACTION_CLEAN_STACK,
    ACTION_TRIGGER_SEARCH,
    ACTION_TRIGGER_CHITCHAT,
    ACTION_RESET_ROUTING,
    ACTION_HANGUP,
    ACTION_REPEAT_BOT_MESSAGES,
    ACTION_CONTINUE_INTERRUPTED_FLOW,
    ACTION_CANCEL_INTERRUPTED_FLOWS,
]

ACTION_SHOULD_SEND_DOMAIN = "send_domain"

# rules allow setting a value of slots or active_loops to None;
# generator substitutes `None`s with this constant to notify rule policy that
# a value should not be set during prediction to activate a rule
SHOULD_NOT_BE_SET = "should_not_be_set"

PREVIOUS_ACTION = "prev_action"
ACTIVE_LOOP = "active_loop"
LOOP_INTERRUPTED = "is_interrupted"
LOOP_REJECTED = "rejected"
TRIGGER_MESSAGE = "trigger_message"
FOLLOWUP_ACTION = "followup_action"
ACTIVE_FLOW = "active_flow"

# start of special user message section
EXTERNAL_MESSAGE_PREFIX = "EXTERNAL: "
# Key to access data in the event metadata
# It specifies if an event was caused by an external entity (e.g. a sensor).
IS_EXTERNAL = "is_external"

ACTION_NAME_SENDER_ID_CONNECTOR_STR = "__sender_id:"

REQUESTED_SLOT = "requested_slot"
FLOW_HASHES_SLOT = "flow_hashes"

FLOW_SLOT_NAMES = [FLOW_HASHES_SLOT]

# slots for audio timeout
GLOBAL_SILENCE_TIMEOUT_KEY = "global_silence_timeout"
SILENCE_TIMEOUT_CHANNEL_KEY = "silence_timeout"
SILENCE_TIMEOUT_SLOT = "silence_timeout"
SLOT_CONSECUTIVE_SILENCE_TIMEOUTS = "consecutive_silence_timeouts"
GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE = 7.0
SILENCE_SLOTS = {SILENCE_TIMEOUT_SLOT, SLOT_CONSECUTIVE_SILENCE_TIMEOUTS}
# slots for knowledge base
SLOT_LISTED_ITEMS = "knowledge_base_listed_objects"
SLOT_LAST_OBJECT = "knowledge_base_last_object"
SLOT_LAST_OBJECT_TYPE = "knowledge_base_last_object_type"
DEFAULT_KNOWLEDGE_BASE_ACTION = "action_query_knowledge_base"

KNOWLEDGE_BASE_SLOT_NAMES = {
    SLOT_LISTED_ITEMS,
    SLOT_LAST_OBJECT,
    SLOT_LAST_OBJECT_TYPE,
}

DEFAULT_SLOT_NAMES = {
    REQUESTED_SLOT,
    SESSION_START_METADATA_SLOT,
    FLOW_HASHES_SLOT,
    MOCKED_DATETIME_SLOT,
}

SLOTS_EXCLUDED_FOR_AGENT = (
    SILENCE_SLOTS | DEFAULT_SLOT_NAMES | KNOWLEDGE_BASE_SLOT_NAMES
)

SLOT_MAPPINGS = "mappings"
MAPPING_CONDITIONS = "conditions"
KEY_MAPPING_TYPE = "type"
KEY_ALLOW_NLU_CORRECTION = "allow_nlu_correction"
KEY_ACTION = "action"
KEY_RUN_ACTION_EVERY_TURN = "run_action_every_turn"
KEY_COEXISTENCE_SYSTEM = "coexistence_system"


class SlotMappingType(Enum):
    """Slot mapping types."""

    FROM_ENTITY = "from_entity"
    FROM_INTENT = "from_intent"
    FROM_TRIGGER_INTENT = "from_trigger_intent"
    FROM_TEXT = "from_text"
    FROM_LLM = "from_llm"
    CONTROLLED = "controlled"

    def __str__(self) -> str:
        """Returns the string representation that should be used in config files."""
        return self.value

    def is_predefined_type(self) -> bool:
        """Returns True if the mapping type is NLU-predefined."""
        return not (
            self == SlotMappingType.CONTROLLED or self == SlotMappingType.FROM_LLM
        )


class SetSlotExtractor(Enum):
    """The extractors that can set a slot."""

    LLM = "LLM"
    COMMAND_PAYLOAD_READER = "CommandPayloadReader"
    NLU = "NLU"
    CUSTOM = "CUSTOM"

    def __str__(self) -> str:
        return self.value


# the keys for `State` (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# represent the origin of a `SubState`
USER = "user"
BOT = "bot"
SLOTS = "slots"

USE_TEXT_FOR_FEATURIZATION = "use_text_for_featurization"
ENTITY_LABEL_SEPARATOR = "#"

RULE_ONLY_SLOTS = "rule_only_slots"
RULE_ONLY_LOOPS = "rule_only_loops"

# if you add more policy/classifier names, make sure to add a test as well to ensure
# that the name and the class stay in sync
POLICY_NAME_TWO_STAGE_FALLBACK = "TwoStageFallbackPolicy"
POLICY_NAME_MAPPING = "MappingPolicy"
POLICY_NAME_FALLBACK = "FallbackPolicy"
POLICY_NAME_FORM = "FormPolicy"
POLICY_NAME_RULE = "RulePolicy"

CLASSIFIER_NAME_FALLBACK = "FallbackClassifier"

POLICIES_THAT_EXTRACT_ENTITIES = {"TEDPolicy"}

ERROR_CODE_KEY = "error_code"
