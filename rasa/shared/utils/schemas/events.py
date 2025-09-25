ENTITIES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "entity": {"type": "string"},
            "confidence": {"type": "number"},
            "extractor": {"type": ["string", "null"]},
            "value": {},
            "role": {"type": ["string", "null"]},
            "group": {"type": ["string", "null"]},
        },
        "required": ["entity", "value"],
    },
}

INTENT = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "confidence": {"type": "number"}},
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "responses": {
            "type": "array",
            "items": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
        "response_templates": {
            "type": "array",
            "items": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
        "confidence": {"type": "number"},
        "intent_response_key": {"type": "string"},
        "utter_action": {"type": "string"},
        "template_name": {"type": "string"},
    },
}

RANKING_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "number"},
            "confidence": {"type": "number"},
            "intent_response_key": {"type": "string"},
        },
    },
}

USER_UTTERED = {
    "properties": {
        "event": {"const": "user"},
        "text": {"type": ["string", "null"]},
        "input_channel": {"type": ["string", "null"]},
        "message_id": {"type": ["string", "null"]},
        "parse_data": {
            "type": "object",
            "properties": {
                "text": {"type": ["string", "null"]},
                "intent_ranking": {"type": "array", "items": INTENT},
                "intent": INTENT,
                "entities": ENTITIES_SCHEMA,
                "response_selector": {
                    "type": "object",
                    "oneOf": [
                        {"properties": {"all_retrieval_intents": {"type": "array"}}},
                        {
                            "patternProperties": {
                                "[\\w/]": {
                                    "type": "object",
                                    "properties": {
                                        "response": RESPONSE_SCHEMA,
                                        "ranking": RANKING_SCHEMA,
                                    },
                                }
                            }
                        },
                    ],
                },
            },
        },
    }
}

ACTION_EXECUTED = {
    "properties": {
        "event": {"const": "action"},
        "policy": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"]},
        "name": {"type": ["string", "null"]},
        "hide_rule_turn": {"type": "boolean"},
        "action_text": {"type": ["string", "null"]},
    }
}

SLOT_SET = {
    "properties": {"event": {"const": "slot"}, "name": {"type": "string"}, "value": {}},
    "required": ["name", "value"],
}

ENTITIES_ADDED = {
    "properties": {"event": {"const": "entities"}, "entities": ENTITIES_SCHEMA},
    "required": ["entities"],
}

USER_UTTERED_FEATURIZATION = {"properties": {"event": {"const": "user_featurization"}}}
REMINDER_CANCELLED = {"properties": {"event": {"const": "cancel_reminder"}}}
REMINDER_SCHEDULED = {"properties": {"event": {"const": "reminder"}}}
ACTION_EXECUTION_REJECTED = {
    "properties": {"event": {"const": "action_execution_rejected"}}
}
FORM_VALIDATION = {"properties": {"event": {"const": "form_validation"}}}
LOOP_INTERRUPTED = {"properties": {"event": {"const": "loop_interrupted"}}}
FORM = {"properties": {"event": {"const": "form"}}}
ACTIVE_LOOP = {"properties": {"event": {"const": "active_loop"}}}
ALL_SLOTS_RESET = {"properties": {"event": {"const": "reset_slots"}}}
CONVERSATION_RESUMED = {"properties": {"event": {"const": "resume"}}}
CONVERSATION_PAUSED = {"properties": {"event": {"const": "pause"}}}
FOLLOWUP_ACTION = {"properties": {"event": {"const": "followup"}}}
STORY_EXPORTED = {"properties": {"event": {"const": "export"}}}
RESTARTED = {"properties": {"event": {"const": "restart"}}}
ACTION_REVERTED = {"properties": {"event": {"const": "undo"}}}
USER_UTTERANCE_REVERTED = {"properties": {"event": {"const": "rewind"}}}
BOT_UTTERED = {"properties": {"event": {"const": "bot"}}}
SESSION_STARTED = {"properties": {"event": {"const": "session_started"}}}
SESSION_ENDED = {"properties": {"event": {"const": "session_ended"}}}
AGENT_UTTERED = {"properties": {"event": {"const": "agent"}}}
FLOW_STARTED = {
    "properties": {"event": {"const": "flow_started"}, "flow_id": {"type": "string"}}
}
FLOW_INTERRUPTED = {
    "properties": {
        "event": {"const": "flow_interrupted"},
        "flow_id": {"type": "string"},
        "step_id": {"type": "string"},
    }
}
FLOW_RESUMED = {
    "properties": {
        "event": {"const": "flow_resumed"},
        "flow_id": {"type": "string"},
        "step_id": {"type": "string"},
    }
}
FLOW_COMPLETED = {
    "properties": {
        "event": {"const": "flow_completed"},
        "flow_id": {"type": "string"},
        "step_id": {"type": "string"},
    }
}
FLOW_CANCELLED = {
    "properties": {
        "event": {"const": "flow_cancelled"},
        "flow_id": {"type": "string"},
        "step_id": {"type": "string"},
    }
}
AGENT_STARTED = {
    "properties": {
        "event": {"const": "agent_started"},
        "agent_id": {"type": "string"},
        "flow_id": {"type": "string"},
    }
}
AGENT_COMPLETED = {
    "properties": {
        "event": {"const": "agent_completed"},
        "agent_id": {"type": "string"},
        "flow_id": {"type": "string"},
        "status": {"type": "string"},
    }
}
AGENT_INTERRUPTED = {
    "properties": {
        "event": {"const": "agent_interrupted"},
        "agent_id": {"type": "string"},
        "flow_id": {"type": "string"},
    }
}
AGENT_RESUMED = {
    "properties": {
        "event": {"const": "agent_resumed"},
        "agent_id": {"type": "string"},
        "flow_id": {"type": "string"},
    }
}
AGENT_CANCELLED = {
    "properties": {
        "event": {"const": "agent_cancelled"},
        "agent_id": {"type": "string"},
        "flow_id": {"type": "string"},
        "reason": {"type": "string"},
    }
}
DIALOGUE_STACK_UPDATED = {
    "properties": {"event": {"const": "stack"}, "update": {"type": "string"}}
}
ROUTING_SESSION_ENDED = {"properties": {"event": {"const": "routing_session_ended"}}}

ERROR_HANDLED = {"properties": {"event": {"const": "error"}}}

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event": {"type": "string"},
        "timestamp": {"type": ["number", "null"]},
        "metadata": {"type": ["object", "null"]},
    },
    "required": ["event"],
    "oneOf": [
        USER_UTTERED,
        ACTION_EXECUTED,
        SLOT_SET,
        ENTITIES_ADDED,
        USER_UTTERED_FEATURIZATION,
        REMINDER_CANCELLED,
        REMINDER_SCHEDULED,
        ACTION_EXECUTION_REJECTED,
        FORM_VALIDATION,
        LOOP_INTERRUPTED,
        FORM,
        ACTIVE_LOOP,
        ALL_SLOTS_RESET,
        CONVERSATION_RESUMED,
        CONVERSATION_PAUSED,
        FOLLOWUP_ACTION,
        STORY_EXPORTED,
        RESTARTED,
        ACTION_REVERTED,
        USER_UTTERANCE_REVERTED,
        BOT_UTTERED,
        SESSION_STARTED,
        AGENT_UTTERED,
        FLOW_STARTED,
        FLOW_INTERRUPTED,
        FLOW_RESUMED,
        FLOW_COMPLETED,
        FLOW_CANCELLED,
        AGENT_STARTED,
        AGENT_COMPLETED,
        AGENT_INTERRUPTED,
        AGENT_RESUMED,
        AGENT_CANCELLED,
        DIALOGUE_STACK_UPDATED,
        ROUTING_SESSION_ENDED,
        SESSION_ENDED,
        ERROR_HANDLED,
    ],
}

EVENTS_SCHEMA = {"type": "array", "items": EVENT_SCHEMA}
