STRING_ARRAY_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "uniqueItems": True,
}

PROJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "nlu": STRING_ARRAY_SCHEMA,
        "rules": STRING_ARRAY_SCHEMA,
        "stories": STRING_ARRAY_SCHEMA,
        "config": {"type": "string"},
        "domain": STRING_ARRAY_SCHEMA,
        "models": {"type": "string"},
        "actions": {"type": "string"},
        "test_data": STRING_ARRAY_SCHEMA,
        "train_test_split": {"type": "string"},
        "results": {"type": "string"},
        "importers": {
            "type": "array",
            "items": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    },
    "required": [
        "version",
        "nlu",
        "rules",
        "stories",
        "config",
        "domain",
        "models",
        "actions",
        "test_data",
        "train_test_split",
        "results",
    ],
    "additionalProperties": False,
}
