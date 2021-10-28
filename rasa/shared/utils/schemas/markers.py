MARKER_ID_SCHEMA = {
    "description": "The unique identifier for a marker.",
    "type": "string",
}

OPERATOR_SCHEMA = {
    "description": "The operator to apply to the list of conditions.",
    "type": "string",
    "enum": ["AND", "AT_LEAST_ONE_NOT", "OR", "NOT", "SEQ", "OCCUR", "NEVER"],
    "default": "AND",
}

EVENT_SCHEMA = {
    "description": "A list of events nested under a condition.",
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "uniqueItems": True,
}

CONDITION_SCHEMA = {
    "description": "The list of conditions for a marker",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "slot_is_set": EVENT_SCHEMA,
            "slot_is_not_set": EVENT_SCHEMA,
            "action_executed": EVENT_SCHEMA,
            "action_not_executed": EVENT_SCHEMA,
            "intent_detected": EVENT_SCHEMA,
            "intent_not_detected": EVENT_SCHEMA,
            "user_uttered": EVENT_SCHEMA,
            "bot_uttered": EVENT_SCHEMA,
        },
        "additionalProperties": False,
    },
    "minItems": 1,
}

MARKERS_SCHEMA = {
    "type": "object",
    "properties": {
        "markers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "marker": MARKER_ID_SCHEMA,
                    "operator": OPERATOR_SCHEMA,
                    "condition": CONDITION_SCHEMA,
                },
                "required": ["marker", "condition"],
            },
        },
    },
    "required": ["markers"],
}
