from typing import Dict, Text, Any


def entity_dict_schema() -> Dict[Text, Any]:
    """Returns: schema for defining entities in Markdown format."""
    return {
        "type": "object",
        "properties": _common_entity_properties(),
        "required": ["entity"],
    }


def _common_entity_properties() -> Dict[Text, Any]:
    return {
        "entity": {"type": "string"},
        "role": {"type": "string"},
        "group": {"type": "string"},
        "value": {"type": "string"},
    }


def rasa_nlu_data_schema() -> Dict[Text, Any]:
    """Returns: schema of the Rasa NLU data format (json format)."""
    entity_properties = _common_entity_properties()
    entity_properties["start"] = {"type": "number"}
    entity_properties["end"] = {"type": "number"}

    training_example_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "intent": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": entity_properties,
                    "required": ["start", "end", "entity"],
                },
            },
        },
        "required": ["text"],
    }

    regex_feature_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "pattern": {"type": "string"}},
    }

    lookup_table_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "elements": {
                "oneOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "string"},
                ]
            },
        },
    }

    return {
        "type": "object",
        "properties": {
            "rasa_nlu_data": {
                "type": "object",
                "properties": {
                    "regex_features": {"type": "array", "items": regex_feature_schema},
                    "common_examples": {
                        "type": "array",
                        "items": training_example_schema,
                    },
                    "intent_examples": {
                        "type": "array",
                        "items": training_example_schema,
                    },
                    "entity_examples": {
                        "type": "array",
                        "items": training_example_schema,
                    },
                    "lookup_tables": {"type": "array", "items": lookup_table_schema},
                },
            }
        },
        "additionalProperties": False,
    }
