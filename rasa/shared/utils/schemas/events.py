ENTITIES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "entity": {"type": "string"},
            "confidence": {"type": "number"},
            "extractor": {},
            "value": {},
            "role": {"type": ["string", "null"]},
            "group": {"type": ["string", "null"]},
        },
    },
}

EVENTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "event": {"type": "string"},
            "timestamp": {"type": ["number", "null"]},
            "metadata": {"type": ["object", "null"]},
        },
        "required": ["event"],
        "oneOf": [
            {
                "properties": {
                    "event": {"const": "user"},
                    "text": {"type": ["string", "null"]},
                    "input_channel": {"type": ["string", "null"]},
                    "message_id": {"type": ["string", "null"]},
                    "parse_data": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "intent_ranking": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "confidence": {"type": "number"},
                                    },
                                },
                            },
                            "intent": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                            },
                            "entities": ENTITIES_SCHEMA,
                        },
                    },
                }
            },
            {
                "properties": {
                    "event": {"const": "action"},
                    "policy": {"type": ["string", "null"]},
                    "confidence": {"type": ["number", "null"]},
                    "name": {"type": ["string", "null"]},
                    "hide_rule_turn": {"type": "boolean"},
                    "action_text": {"type": ["string", "null"]},
                }
            },
            {
                "properties": {
                    "event": {"const": "slot"},
                    "name": {"type": "string"},
                    "value": {},
                }
            },
            {
                "properties": {
                    "event": {"const": "entities"},
                    "entities": ENTITIES_SCHEMA,
                }
            },
            {"properties": {"event": {"const": "user_featurization"}}},
            {"properties": {"event": {"const": "cancel_reminder"}}},
            {"properties": {"event": {"const": "reminder"}}},
            {"properties": {"event": {"const": "action_execution_rejected"}}},
            {"properties": {"event": {"const": "form_validation"}}},
            {"properties": {"event": {"const": "loop_interrupted"}}},
            {"properties": {"event": {"const": "form"}}},
            {"properties": {"event": {"const": "active_loop"}}},
            {"properties": {"event": {"const": "reset_slots"}}},
            {"properties": {"event": {"const": "resume"}}},
            {"properties": {"event": {"const": "pause"}}},
            {"properties": {"event": {"const": "followup"}}},
            {"properties": {"event": {"const": "export"}}},
            {"properties": {"event": {"const": "restart"}}},
            {"properties": {"event": {"const": "undo"}}},
            {"properties": {"event": {"const": "rewind"}}},
            {"properties": {"event": {"const": "bot"}}},
            {"properties": {"event": {"const": "session_started"}}},
            {"properties": {"event": {"const": "agent"}}},
        ],
    },
}
