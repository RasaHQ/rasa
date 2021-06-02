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
    },
}

INTENT = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "confidence": {"type": "number"}},
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
                            "intent_ranking": {"type": "array", "items": INTENT},
                            "intent": INTENT,
                            "entities": ENTITIES_SCHEMA,
                            "response_selector": {
                                "type": "object",
                                "properties": {
                                    "all_retrieval_intents": {"type": "array"},
                                    "default": {
                                        "type": "object",
                                        "properties": {
                                            "response": {
                                                "type": "object",
                                                "properties": {
                                                    "responses": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "text": {
                                                                    "type": "string"
                                                                }
                                                            },
                                                        },
                                                    },
                                                    "response_templates": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "text": {
                                                                    "type": "string"
                                                                }
                                                            },
                                                        },
                                                    },
                                                    "confidence": {"type": "number"},
                                                    "intent_response_key": {
                                                        "type": "string"
                                                    },
                                                    "utter_action": {"type": "string"},
                                                    "template_name": {"type": "string"},
                                                },
                                            },
                                            "ranking": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "confidence": {
                                                            "type": "number"
                                                        },
                                                        "intent_response_key": {
                                                            "type": "string"
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
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
                },
                "required": ["name", "value"],
            },
            {
                "properties": {
                    "event": {"const": "entities"},
                    "entities": ENTITIES_SCHEMA,
                },
                "required": ["entities"],
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
